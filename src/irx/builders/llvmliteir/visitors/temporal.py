# mypy: disable-error-code=no-redef

"""
title: Temporal literal visitor mixins for llvmliteir.
"""

from datetime import datetime
from datetime import time as time_value

from llvmlite import ir

from irx import astx
from irx.builders.llvmliteir.core import VisitorCore
from irx.builders.llvmliteir.protocols import VisitorMixinBase


class TemporalVisitorMixin(VisitorMixinBase):
    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralTime) -> None:
        """
        title: Visit LiteralTime nodes.
        parameters:
          node:
            type: astx.LiteralTime
        """
        hour_minute_count = 2
        hour_minute_second_count = 3
        max_hour = 23
        max_minute_second = 59

        value = node.value.strip()
        parts = value.split(":")
        if len(parts) not in (hour_minute_count, hour_minute_second_count):
            raise Exception(
                f"LiteralTime: invalid time format '{node.value}'. "
                "Expected 'HH:MM' or 'HH:MM:SS'."
            )

        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except Exception as exc:
            raise Exception(
                f"LiteralTime: invalid hour/minute in '{node.value}'."
            ) from exc

        if len(parts) == hour_minute_second_count:
            sec_part = parts[2]
            if "." in sec_part:
                raise Exception(
                    "LiteralTime: fractional seconds "
                    f"not supported in '{node.value}'."
                )
            try:
                second = int(sec_part)
            except Exception as exc:
                raise Exception(
                    f"LiteralTime: invalid seconds in '{node.value}'."
                ) from exc
        else:
            second = 0

        if not (0 <= hour <= max_hour):
            raise Exception(
                f"LiteralTime: hour out of range in '{node.value}'."
            )
        if not (0 <= minute <= max_minute_second):
            raise Exception(
                f"LiteralTime: minute out of range in '{node.value}'."
            )
        if not (0 <= second <= max_minute_second):
            raise Exception(
                f"LiteralTime: second out of range in '{node.value}'."
            )

        i32 = self._llvm.INT32_TYPE
        const_time = ir.Constant(
            self._llvm.TIME_TYPE,
            [
                ir.Constant(i32, hour),
                ir.Constant(i32, minute),
                ir.Constant(i32, second),
            ],
        )
        self.result_stack.append(const_time)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralTimestamp) -> None:
        """
        title: Visit LiteralTimestamp nodes.
        parameters:
          node:
            type: astx.LiteralTimestamp
        """
        nanosecond_digits = 9
        max_hour = 23
        max_minute_second = 59

        value = node.value.strip()
        if "T" in value:
            date_part, time_part = value.split("T", 1)
        elif " " in value:
            date_part, time_part = value.split(" ", 1)
        else:
            raise Exception(
                "LiteralTimestamp: invalid format '"
                f"{node.value}'. Expected 'YYYY-MM-DDTHH:MM:SS"
                "[.fffffffff]' (or space instead of 'T')."
            )

        if time_part.endswith("Z") or "+" in time_part or "-" in time_part[2:]:
            raise Exception(
                "LiteralTimestamp: timezone offsets not supported in '"
                f"{node.value}'."
            )

        try:
            y_str, m_str, d_str = date_part.split("-")
            year = int(y_str)
            month = int(m_str)
            day = int(d_str)
            datetime(year, month, day)
        except ValueError as exc:
            raise Exception(
                "LiteralTimestamp: invalid date in '"
                f"{node.value}'. Expected valid 'YYYY-MM-DD'."
            ) from exc
        except Exception as exc:
            raise Exception(
                "LiteralTimestamp: invalid date part in '"
                f"{node.value}'. Expected 'YYYY-MM-DD'."
            ) from exc

        frac_ns = 0
        try:
            if "." in time_part:
                hms, frac = time_part.split(".", 1)
                if not frac.isdigit():
                    raise ValueError("fractional seconds must be digits")
                if len(frac) > nanosecond_digits:
                    frac = frac[:nanosecond_digits]
                frac_ns = int(frac.ljust(nanosecond_digits, "0"))
            else:
                hms = time_part

            h_str, m_str, s_str = hms.split(":")
            hour = int(h_str)
            minute = int(m_str)
            second = int(s_str)
        except Exception as exc:
            raise Exception(
                "LiteralTimestamp: invalid time part in '"
                f"{node.value}'. Expected 'HH:MM:SS'"
                " (optionally with '.fffffffff')."
            ) from exc

        if not (0 <= hour <= max_hour):
            raise Exception(
                f"LiteralTimestamp: hour out of range in '{node.value}'."
            )
        if not (0 <= minute <= max_minute_second):
            raise Exception(
                f"LiteralTimestamp: minute out of range in '{node.value}'."
            )
        if not (0 <= second <= max_minute_second):
            raise Exception(
                f"LiteralTimestamp: second out of range in '{node.value}'."
            )

        i32 = self._llvm.INT32_TYPE
        const_ts = ir.Constant(
            self._llvm.TIMESTAMP_TYPE,
            [
                ir.Constant(i32, year),
                ir.Constant(i32, month),
                ir.Constant(i32, day),
                ir.Constant(i32, hour),
                ir.Constant(i32, minute),
                ir.Constant(i32, second),
                ir.Constant(i32, frac_ns),
            ],
        )
        self.result_stack.append(const_ts)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralDateTime) -> None:
        """
        title: Visit LiteralDateTime nodes.
        parameters:
          node:
            type: astx.LiteralDateTime
        """
        hour_minute_count = 2
        hour_minute_second_count = 3
        max_hour = 23
        max_minute_second = 59

        value = node.value.strip()
        if "T" in value:
            date_part, time_part = value.split("T", 1)
        elif " " in value:
            date_part, time_part = value.split(" ", 1)
        else:
            raise ValueError(
                f"LiteralDateTime: invalid format '{node.value}'. "
                "Expected 'YYYY-MM-DDTHH:MM[:SS]' (or space instead of 'T')."
            )

        if "." in time_part:
            raise ValueError(
                "LiteralDateTime: fractional seconds not supported in "
                f"'{node.value}'. Use LiteralTimestamp instead."
            )
        if time_part.endswith("Z") or "+" in time_part or "-" in time_part[2:]:
            raise ValueError(
                "LiteralDateTime: timezone offsets not supported in "
                f"'{node.value}'. Use LiteralTimestamp for timezones."
            )

        try:
            y_str, m_str, d_str = date_part.split("-")
            year = int(y_str)
            month = int(m_str)
            day = int(d_str)
        except Exception as exc:
            raise ValueError(
                f"LiteralDateTime: invalid date part in '{node.value}'. "
                "Expected 'YYYY-MM-DD'."
            ) from exc

        int32_min, int32_max = -(2**31), 2**31 - 1
        if not (int32_min <= year <= int32_max):
            raise ValueError(
                f"LiteralDateTime: year out of 32-bit range in '{node.value}'."
            )

        try:
            parts = time_part.split(":")
            if len(parts) not in (hour_minute_count, hour_minute_second_count):
                raise ValueError("time must be HH:MM or HH:MM:SS")
            hour = int(parts[0])
            minute = int(parts[1])
            second = (
                int(parts[2]) if len(parts) == hour_minute_second_count else 0
            )
        except Exception as exc:
            raise ValueError(
                f"LiteralDateTime: invalid time part in '{node.value}'. "
                "Expected 'HH:MM' or 'HH:MM:SS'."
            ) from exc

        if not (0 <= hour <= max_hour):
            raise ValueError(
                f"LiteralDateTime: hour out of range in '{node.value}'."
            )
        if not (0 <= minute <= max_minute_second):
            raise ValueError(
                f"LiteralDateTime: minute out of range in '{node.value}'."
            )
        if not (0 <= second <= max_minute_second):
            raise ValueError(
                f"LiteralDateTime: second out of range in '{node.value}'."
            )

        try:
            datetime(year, month, day)
            time_value(hour, minute, second)
        except ValueError as exc:
            raise ValueError(
                "LiteralDateTime: invalid calendar date/time in "
                f"'{node.value}'."
            ) from exc

        i32 = self._llvm.INT32_TYPE
        const_dt = ir.Constant(
            self._llvm.DATETIME_TYPE,
            [
                ir.Constant(i32, year),
                ir.Constant(i32, month),
                ir.Constant(i32, day),
                ir.Constant(i32, hour),
                ir.Constant(i32, minute),
                ir.Constant(i32, second),
            ],
        )
        self.result_stack.append(const_dt)
