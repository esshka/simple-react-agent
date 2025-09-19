"""
Comprehensive test runner for all test modules in the tests directory.

This module provides functions to run individual test modules or all tests
with proper error handling and reporting.
"""

import os
import sys
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class TestRunner:
    """A comprehensive test runner for the project."""

    def __init__(self, tests_dir: str = "tests"):
        """Initialize the test runner.

        Args:
            tests_dir: Directory containing test files
        """
        self.tests_dir = Path(tests_dir)
        self.project_root = Path(__file__).parent.parent
        self.test_files = self._discover_test_files()
        self.results = {}

    def _discover_test_files(self) -> List[Path]:
        """Discover all test files in the tests directory."""
        test_files = []

        # Find all Python test files
        for py_file in self.tests_dir.glob("test_*.py"):
            if py_file.name != "test_runner.py":  # Exclude this file
                test_files.append(py_file)

        return sorted(test_files)

    def _get_python_path(self) -> str:
        """Get the correct PYTHONPATH for running tests."""
        src_path = self.project_root / "src"
        return str(src_path)

    def _run_command(self, command: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
        """Run a command and return return code, stdout, and stderr."""
        env = os.environ.copy()
        env["PYTHONPATH"] = self._get_python_path()

        if cwd is None:
            cwd = str(self.project_root)

        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                env=env,
                timeout=300  # 5 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Test execution timed out"
        except Exception as e:
            return -1, "", f"Error running command: {e}"

    def run_test_with_pytest(self, test_file: Path) -> Dict[str, Any]:
        """Run a specific test file using pytest."""
        try:
            return_code, stdout, stderr = self._run_command([
                sys.executable, "-m", "pytest", str(test_file), "-v"
            ])

            return {
                "file": test_file.name,
                "runner": "pytest",
                "success": return_code == 0,
                "return_code": return_code,
                "stdout": stdout,
                "stderr": stderr,
                "total_tests": self._count_pytest_tests(stdout),
                "passed_tests": self._count_pytest_passed(stdout),
                "failed_tests": self._count_pytest_failed(stdout)
            }
        except Exception as e:
            return {
                "file": test_file.name,
                "runner": "pytest",
                "success": False,
                "error": str(e)
            }

    def run_test_with_unittest(self, test_file: Path) -> Dict[str, Any]:
        """Run a specific test file using unittest."""
        try:
            return_code, stdout, stderr = self._run_command([
                sys.executable, str(test_file)
            ])

            return {
                "file": test_file.name,
                "runner": "unittest",
                "success": return_code == 0,
                "return_code": return_code,
                "stdout": stdout,
                "stderr": stderr
            }
        except Exception as e:
            return {
                "file": test_file.name,
                "runner": "unittest",
                "success": False,
                "error": str(e)
            }

    def _count_pytest_tests(self, stdout: str) -> int:
        """Count total tests from pytest output."""
        import re
        match = re.search(r'(\d+)\s+test', stdout)
        return int(match.group(1)) if match else 0

    def _count_pytest_passed(self, stdout: str) -> int:
        """Count passed tests from pytest output."""
        import re
        match = re.search(r'(\d+)\s+passed', stdout)
        return int(match.group(1)) if match else 0

    def _count_pytest_failed(self, stdout: str) -> int:
        """Count failed tests from pytest output."""
        import re
        match = re.search(r'(\d+)\s+failed', stdout)
        return int(match.group(1)) if match else 0

    def run_individual_test(self, test_file: Path) -> Dict[str, Any]:
        """Run an individual test file using the best available runner."""
        # Try pytest first, fallback to unittest
        result = self.run_test_with_pytest(test_file)

        if result["success"] or "error" not in result:
            return result

        # Fallback to unittest
        return self.run_test_with_unittest(test_file)

    def run_maths_tests(self) -> Dict[str, Any]:
        """Run mathematics tools tests."""
        test_file = self.tests_dir / "test_maths_tools.py"
        if test_file.exists():
            return self.run_individual_test(test_file)
        return {"error": "Maths test file not found", "file": "test_maths_tools.py"}

    def run_mongo_tests(self) -> Dict[str, Any]:
        """Run MongoDB tools tests."""
        test_file = self.tests_dir / "test_mongo_tools.py"
        if test_file.exists():
            return self.run_individual_test(test_file)
        return {"error": "MongoDB test file not found", "file": "test_mongo_tools.py"}

    def run_wiki_tests(self) -> Dict[str, Any]:
        """Run Wikipedia tools tests."""
        test_file = self.tests_dir / "test_wiki_tools.py"
        if test_file.exists():
            return self.run_individual_test(test_file)
        return {"error": "Wikipedia test file not found", "file": "test_wiki_tools.py"}

    def run_web_search_tests(self) -> Dict[str, Any]:
        """Run web search tools tests."""
        test_file = self.tests_dir / "test_web_search_tools.py"
        if test_file.exists():
            return self.run_individual_test(test_file)
        return {"error": "Web search test file not found", "file": "test_web_search_tools.py"}

    def run_searxng_tests(self) -> Dict[str, Any]:
        """Run SearXNG tools tests."""
        test_file = self.tests_dir / "test_searxng_tools.py"
        if test_file.exists():
            return self.run_individual_test(test_file)
        return {"error": "SearXNG test file not found", "file": "test_searxng_tools.py"}

    def run_ai_agent_integration_tests(self) -> Dict[str, Any]:
        """Run AI agent integration tests."""
        test_file = self.tests_dir / "test_ai_agent_integration.py"
        if test_file.exists():
            return self.run_individual_test(test_file)
        return {"error": "AI agent integration test file not found", "file": "test_ai_agent_integration.py"}

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all discovered tests."""
        print("🚀 Running all tests...")
        print(f"Discovered {len(self.test_files)} test files:")

        all_results = {}
        total_passed = 0
        total_failed = 0

        for test_file in self.test_files:
            print(f"\n📁 Running {test_file.name}...")
            result = self.run_individual_test(test_file)
            all_results[test_file.name] = result

            if result.get("success", False):
                print(f"✅ {test_file.name}: PASSED")
                total_passed += 1
            else:
                print(f"❌ {test_file.name}: FAILED")
                total_failed += 1

                if "error" in result:
                    print(f"   Error: {result['error']}")

        # Summary
        print(f"\n📊 Test Summary:")
        print(f"   Total test files: {len(self.test_files)}")
        print(f"   Passed: {total_passed}")
        print(f"   Failed: {total_failed}")

        self.results = all_results
        return {
            "total_files": len(self.test_files),
            "passed": total_passed,
            "failed": total_failed,
            "results": all_results
        }

    def run_core_tools_tests(self) -> Dict[str, Any]:
        """Run only the core tools tests (excluding AI integration)."""
        core_tests = [
            "test_maths_tools.py",
            "test_mongo_tools.py",
            "test_wiki_tools.py",
            "test_web_search_tools.py",
            "test_searxng_tools.py"
        ]

        print("🔧 Running core tools tests...")
        results = {}
        passed = 0
        failed = 0

        for test_name in core_tests:
            test_file = self.tests_dir / test_name
            if test_file.exists():
                print(f"\n📁 Running {test_name}...")
                result = self.run_individual_test(test_file)
                results[test_name] = result

                if result.get("success", False):
                    print(f"✅ {test_name}: PASSED")
                    passed += 1
                else:
                    print(f"❌ {test_name}: FAILED")
                    failed += 1
            else:
                print(f"⚠️  {test_name}: Not found")
                results[test_name] = {"error": "File not found"}
                failed += 1

        print(f"\n📊 Core Tools Summary:")
        print(f"   Total: {passed + failed}")
        print(f"   Passed: {passed}")
        print(f"   Failed: {failed}")

        return {
            "total_files": passed + failed,
            "passed": passed,
            "failed": failed,
            "results": results
        }

    def print_test_results(self, result: Dict[str, Any]) -> None:
        """Print test results in a readable format."""
        print(f"\n📋 Test Results for {result.get('file', 'Unknown')}:")

        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
            return

        if result.get("success", False):
            print(f"   ✅ PASSED")

            if "total_tests" in result:
                print(f"   📊 Total tests: {result['total_tests']}")
                print(f"   ✅ Passed: {result['passed_tests']}")
                print(f"   ❌ Failed: {result['failed_tests']}")
        else:
            print(f"   ❌ FAILED")
            if "stderr" in result and result["stderr"]:
                print(f"   📝 Error output:")
                print(f"      {result['stderr'][:500]}...")  # Show first 500 chars

    def generate_test_report(self) -> str:
        """Generate a comprehensive test report."""
        if not self.results:
            return "No test results available. Run tests first."

        report = ["📊 COMPREHENSIVE TEST REPORT", "=" * 50]

        total_files = len(self.results)
        passed_files = sum(1 for r in self.results.values() if r.get("success", False))
        failed_files = total_files - passed_files

        report.append(f"Total Test Files: {total_files}")
        report.append(f"Passed: {passed_files}")
        report.append(f"Failed: {failed_files}")
        report.append(f"Success Rate: {passed_files/total_files*100:.1f}%")
        report.append("")

        # Individual file results
        report.append("📁 INDIVIDUAL FILE RESULTS:")
        report.append("-" * 30)

        for file_name, result in self.results.items():
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            report.append(f"{file_name}: {status}")

            if "total_tests" in result:
                total = result['total_tests']
                passed = result['passed_tests']
                failed = result['failed_tests']
                report.append(f"   Tests: {total} total, {passed} passed, {failed} failed")

        # Summary by category
        report.append("")
        report.append("📂 SUMMARY BY CATEGORY:")
        report.append("-" * 30)

        categories = {
            "Mathematics": "test_maths_tools.py",
            "MongoDB": "test_mongo_tools.py",
            "Wikipedia": "test_wiki_tools.py",
            "Web Search": "test_web_search_tools.py",
            "SearXNG": "test_searxng_tools.py",
            "AI Integration": "test_ai_agent_integration.py"
        }

        for category, file_name in categories.items():
            if file_name in self.results:
                result = self.results[file_name]
                status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
                report.append(f"{category}: {status}")

        return "\n".join(report)


# Global test runner instance
_test_runner = TestRunner()


# Convenience functions for running tests
def run_maths_tests() -> Dict[str, Any]:
    """Run mathematics tools tests."""
    return _test_runner.run_maths_tests()


def run_mongo_tests() -> Dict[str, Any]:
    """Run MongoDB tools tests."""
    return _test_runner.run_mongo_tests()


def run_wiki_tests() -> Dict[str, Any]:
    """Run Wikipedia tools tests."""
    return _test_runner.run_wiki_tests()


def run_web_search_tests() -> Dict[str, Any]:
    """Run web search tools tests."""
    return _test_runner.run_web_search_tests()


def run_searxng_tests() -> Dict[str, Any]:
    """Run SearXNG tools tests."""
    return _test_runner.run_searxng_tests()


def run_ai_agent_integration_tests() -> Dict[str, Any]:
    """Run AI agent integration tests."""
    return _test_runner.run_ai_agent_integration_tests()


def run_core_tools_tests() -> Dict[str, Any]:
    """Run core tools tests only."""
    return _test_runner.run_core_tools_tests()


def run_all_tests() -> Dict[str, Any]:
    """Run all tests."""
    return _test_runner.run_all_tests()


def print_test_report() -> None:
    """Print comprehensive test report."""
    print(_test_runner.generate_test_report())


def get_test_runner() -> TestRunner:
    """Get the test runner instance for advanced usage."""
    return _test_runner


# List all available test functions
def list_available_tests() -> List[str]:
    """List all available test functions."""
    return [
        "run_maths_tests()",
        "run_mongo_tests()",
        "run_wiki_tests()",
        "run_web_search_tests()",
        "run_searxng_tests()",
        "run_ai_agent_integration_tests()",
        "run_core_tools_tests()",
        "run_all_tests()"
    ]


# Export all functions
__all__ = [
    "TestRunner",
    "run_maths_tests",
    "run_mongo_tests",
    "run_wiki_tests",
    "run_web_search_tests",
    "run_searxng_tests",
    "run_ai_agent_integration_tests",
    "run_core_tools_tests",
    "run_all_tests",
    "print_test_report",
    "get_test_runner",
    "list_available_tests"
]


if __name__ == "__main__":
    """Main execution for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Run tests for the simple-react-agent project")
    parser.add_argument("--maths", action="store_true", help="Run mathematics tests")
    parser.add_argument("--mongo", action="store_true", help="Run MongoDB tests")
    parser.add_argument("--wiki", action="store_true", help="Run Wikipedia tests")
    parser.add_argument("--web-search", action="store_true", help="Run web search tests")
    parser.add_argument("--searxng", action="store_true", help="Run SearXNG tests")
    parser.add_argument("--ai-integration", action="store_true", help="Run AI agent integration tests")
    parser.add_argument("--core", action="store_true", help="Run core tools tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--report", action="store_true", help="Show detailed report")
    parser.add_argument("--list", action="store_true", help="List available test functions")

    args = parser.parse_args()

    if args.list:
        print("Available test functions:")
        for func in list_available_tests():
            print(f"  {func}")
        sys.exit(0)

    runner = TestRunner()

    # Run requested tests
    if args.all:
        result = run_all_tests()
    elif args.core:
        result = run_core_tools_tests()
    elif args.maths:
        result = run_maths_tests()
    elif args.mongo:
        result = run_mongo_tests()
    elif args.wiki:
        result = run_wiki_tests()
    elif args.web_search:
        result = run_web_search_tests()
    elif args.searxng:
        result = run_searxng_tests()
    elif args.ai_integration:
        result = run_ai_agent_integration_tests()
    else:
        # Default: run all tests
        result = run_all_tests()

    # Print results
    if args.report:
        print_test_report()

    # Exit with appropriate code
    sys.exit(0 if result.get("failed", 0) == 0 else 1)