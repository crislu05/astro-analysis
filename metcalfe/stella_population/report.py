"""Shared PASS/FAIL reporting for Metcalfe validation scripts."""


class Report:
    def __init__(self):
        self.number_passed = 0
        self.number_failed = 0

    def check(self, label, value, target, tolerance, unit=""):
        passed = bool(abs(value - target) <= tolerance)
        print(
            f"  [{'PASS' if passed else 'FAIL'}] {label}: got {value:.6g}{unit}, "
            f"expected {target:.6g}{unit} +/- {tolerance:.2g}"
        )
        self.number_passed += passed
        self.number_failed += not passed
        return passed

    def check_true(self, label, condition):
        condition = bool(condition)
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
        self.number_passed += condition
        self.number_failed += not condition
        return condition

    def summary(self):
        print(f"\nSummary: {self.number_passed} passed, {self.number_failed} failed")
        return self.number_failed == 0