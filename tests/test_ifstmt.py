from irx import IfStmt  # Adjust this based on the actual module structure
import unittest
from irx import IfStmt  # Adjust the import based on the actual module structure

class TestIfStmt(unittest.TestCase):
    def test_ifstmt_true_condition(self):
        stmt = IfStmt(lambda: True, lambda: "True branch", lambda: "False branch")
        self.assertEqual(stmt.execute(), "True branch")

    def test_ifstmt_false_condition(self):
        stmt = IfStmt(lambda: False, lambda: "True branch", lambda: "False branch")
        self.assertEqual(stmt.execute(), "False branch")

    def test_ifstmt_no_false_branch(self):
        stmt = IfStmt(lambda: False, lambda: "True branch")
        self.assertIsNone(stmt.execute())

    def test_ifstmt_invalid_condition(self):
        with self.assertRaises(TypeError):
            IfStmt("not a function", lambda: "True branch", lambda: "False branch")

    def test_ifstmt_invalid_true_branch(self):
        with self.assertRaises(TypeError):
            IfStmt(lambda: True, "not a function", lambda: "False branch")

    def test_ifstmt_invalid_false_branch(self):
        with self.assertRaises(TypeError):
            IfStmt(lambda: False, lambda: "True branch", "not a function")

    def test_ifstmt_complex_condition(self):
        stmt = IfStmt(lambda: 5 > 3 and 10 != 2, lambda: "True branch", lambda: "False branch")
        self.assertEqual(stmt.execute(), "True branch")

    def test_ifstmt_no_branches(self):
        with self.assertRaises(TypeError):
            IfStmt(lambda: True)

    def test_ifstmt_condition_raises_exception(self):
        with self.assertRaises(ZeroDivisionError):
            IfStmt(lambda: 1 / 0, lambda: "True branch", lambda: "False branch").execute()

    def test_ifstmt_true_branch_raises_exception(self):
        with self.assertRaises(ZeroDivisionError):
            IfStmt(lambda: True, lambda: 1 / 0, lambda: "False branch").execute()

    def test_ifstmt_false_branch_raises_exception(self):
        with self.assertRaises(ZeroDivisionError):
            IfStmt(lambda: False, lambda: "True branch", lambda: 1 / 0).execute()

if __name__ == "__main__":
    unittest.main()
