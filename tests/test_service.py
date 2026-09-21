import copy
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from opbench import service
from opbench.catalog import expand_template, normalize_case, workload
from opbench.server import make_handler


def report(name="test", latency=20):
    return {"schema_version": 1, "run": {"name": name, "device": {"name": "CPU", "backend": "cpu"},
            "timing_method": "synchronized_wall_per_iteration"}, "results": [
            {"operator": "mm", "params": {"m": 8, "n": 16, "k": 32}, "dtype": "float32", "stage": "forward", "wall_us": latency, "status": "pass"}]}


class DatabaseTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.temp.name)/"test.sqlite3")
        service.initialize(self.db)

    def tearDown(self):
        self.temp.cleanup()


class DomainTests(DatabaseTestCase):
    def test_analytic_workload(self):
        case = normalize_case({"operator": "addbmm", "params": {"batch": 8, "m": 256, "n": 256, "k": 64}, "dtype": "bfloat16"})
        w = workload(case)
        self.assertEqual(w["flops"], 67108864)
        self.assertEqual(w["bytes"], 786432)
        backward = normalize_case({**case, "stage": "backward", "gradient_scope": "all"})
        self.assertEqual(workload(backward)["flops"], 134217728)
        self.assertIsNone(workload(backward)["bytes"])

    def test_transaction_and_reopen(self):
        rid = service.import_report(self.db, report())["id"]
        service.initialize(self.db)
        exported = service.export_report(self.db, rid)
        self.assertEqual(exported["results"][0]["wall_us"], 20)
        self.assertAlmostEqual(exported["results"][0]["tflops"], 8192/20/1e6)
        self.assertEqual(len(service.list_runs(self.db)), 1)
        invalid = report()
        invalid["results"].append({**invalid["results"][0], "wall_us": -1})
        with self.assertRaises(ValueError):
            service.import_report(self.db, invalid)
        self.assertEqual(len(service.list_runs(self.db)), 1)

    def test_pairing_uses_semantics_not_display_name(self):
        left = report("left", 30)
        right = report("right", 10)
        right["results"][0]["name"] = "different display label"
        l = service.import_report(self.db, left)["id"]
        r = service.import_report(self.db, right)["id"]
        rows = service.compare(self.db, l, r)["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["speedup"], 3)
        right["results"][0]["module_mode"] = "train"
        r2 = service.import_report(self.db, right)["id"]
        rows = service.compare(self.db, l, r2)["rows"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(not x["paired"] for x in rows))

    def test_failed_and_missing_never_get_speedup(self):
        failed = report()
        failed["results"][0].update(status="unsupported", wall_us=None)
        l = service.import_report(self.db, report())["id"]
        r = service.import_report(self.db, failed)["id"]
        self.assertIsNone(service.compare(self.db,l,r)["rows"][0]["speedup"])

    def test_archive_restore_export(self):
        rid=service.import_report(self.db,report())["id"]
        service.archive_run(self.db,rid,True)
        self.assertEqual(service.list_runs(self.db),[])
        self.assertEqual(len(service.list_runs(self.db,True)),1)
        self.assertEqual(len(service.export_report(self.db,rid)["results"]),1)
        service.archive_run(self.db,rid,False)
        self.assertEqual(len(service.list_runs(self.db)),1)

    def test_null_backward_metrics_and_no_nan(self):
        r=report()
        r["results"][0]["stage"]="backward"
        out=service.export_report(self.db,service.import_report(self.db,r)["id"])["results"][0]
        self.assertIsNone(out["bandwidth_gbs"])
        for v in (float("nan"), float("inf"), -1, True):
            r=report();r["results"][0]["wall_us"]=v
            with self.assertRaises(ValueError): service.validate_report(r)

    def test_templates_cover_exactly_28_operators(self):
        for name in ("smoke", "standard"):
            template=json.loads(Path(f"templates/{name}.json").read_text(encoding="utf-8"))
            cases=expand_template(template)
            self.assertEqual(len(set(c["operator"] for c in cases)),28)
            self.assertEqual(len(cases),len(set(c["case_key"] for c in cases)))
        bad=copy.deepcopy(template);bad["operators"][0]["params"][0]["m"]=0
        with self.assertRaises(ValueError):expand_template(bad)


class HTTPTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        self.http=ThreadingHTTPServer(("127.0.0.1",0),make_handler(self.db,"test-secret"))
        self.thread=threading.Thread(target=self.http.serve_forever,daemon=True)
        self.thread.start()
        self.url=f"http://127.0.0.1:{self.http.server_port}"

    def tearDown(self):
        self.http.shutdown();self.http.server_close();self.thread.join()
        super().tearDown()

    def request(self,path,body=None,token=True,origin=None):
        headers={"Content-Type":"application/json"}
        if token:headers["Authorization"]="Bearer test-secret"
        if origin:headers["Origin"]=origin
        req=urllib.request.Request(self.url+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
        with urllib.request.urlopen(req) as res:return res.status,json.loads(res.read())

    def test_api_import_compare_and_auth(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:self.request('/api/runs',token=False)
        self.assertEqual(ctx.exception.code,401)
        _,l=self.request('/api/import',report("A",30))
        _,r=self.request('/api/import',report("B",10))
        _,data=self.request(f"/api/compare?left={l['id']}&right={r['id']}")
        self.assertEqual(data["rows"][0]["speedup"],3)
        self.assertEqual(self.request('/api/templates')[0],200)
        with self.assertRaises(urllib.error.HTTPError) as ctx:self.request('/api/import',report(),origin='https://untrusted.example')
        self.assertEqual(ctx.exception.code,403)

    def test_static_traversal_and_validation(self):
        for path in ('/../opbench/server.py','/api/runs/nonexistent'):
            with self.assertRaises(urllib.error.HTTPError) as ctx:self.request(path)
            self.assertEqual(ctx.exception.code,404)
        with self.assertRaises(urllib.error.HTTPError) as ctx:self.request('/api/import',{'schema_version':0})
        self.assertEqual(ctx.exception.code,400)


if __name__ == '__main__': unittest.main()
