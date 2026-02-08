#!/usr/bin/env python3
"""
Raspberry Pi Power Monitor - 树莓派功耗监控模块
================================================
支持 Pi 5 (PMIC 直读) 和 Pi 4B (软件估算 / INA219 硬件)
设计为在 llama.cpp 推理期间并行运行，对主进程影响极小。

用法:
  1. 作为模块导入:
       from rpi_power_monitor import PowerMonitor
       mon = PowerMonitor(interval=1.0)
       mon.start()
       # ... 运行 llama.cpp ...
       mon.stop()
       mon.summary()

  2. 命令行独立运行:
       python3 rpi_power_monitor.py --interval 1 --output power_log.csv

  3. 包裹 llama.cpp 运行:
       python3 rpi_power_monitor.py --wrap "./llama-cli -m model.gguf -p 'Hello'"

作者: Claude (Anthropic)
"""

import subprocess
import re
import time
import threading
import csv
import os
import sys
import signal
import argparse
import statistics
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
from pathlib import Path


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class PowerReading:
    """单次功率读数"""
    timestamp: float              # Unix 时间戳
    total_power_w: float          # 总功率 (瓦)
    cpu_temp_c: float             # CPU 温度 (°C)
    cpu_freq_mhz: int             # CPU 频率 (MHz)
    cpu_usage_percent: float      # CPU 使用率 (%)
    throttled: str                # 节流状态
    method: str                   # 测量方法: "pmic", "ina219", "estimate"
    branches: Optional[Dict[str, float]] = None  # Pi 5 各分支功率明细 (可选)


@dataclass
class PowerSummary:
    """监控汇总"""
    duration_s: float
    samples: int
    method: str
    avg_power_w: float
    min_power_w: float
    max_power_w: float
    std_power_w: float
    avg_temp_c: float
    max_temp_c: float
    avg_cpu_percent: float
    total_energy_wh: float        # 总能耗 (瓦时)
    total_energy_j: float         # 总能耗 (焦耳)


# ============================================================================
# 平台检测
# ============================================================================

def detect_pi_model() -> str:
    """
    检测树莓派型号。
    返回: "pi5", "pi4b", "pi4", "unknown"
    """
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip().rstrip('\x00')
        if "Raspberry Pi 5" in model:
            return "pi5"
        elif "Raspberry Pi 4 Model B" in model:
            return "pi4b"
        elif "Raspberry Pi 4" in model:
            return "pi4"
        else:
            return "unknown"
    except FileNotFoundError:
        return "unknown"


def check_ina219_available() -> bool:
    """检查 INA219 是否通过 I2C 连接 (默认地址 0x40)"""
    try:
        result = subprocess.run(
            ["i2cdetect", "-y", "1"],
            capture_output=True, text=True, timeout=5
        )
        return "40" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ============================================================================
# 功率读取后端
# ============================================================================

class Pi5PMICReader:
    """
    Pi 5 PMIC 功率读取器
    ----------------------
    通过 `vcgencmd pmic_read_adc` 读取 12 条电源分支的 V/I，
    计算 ΣV×I，再经线性校正得到更接近真实值的功率。

    校正公式 (来自 jfikar/RPi5-power 项目，用外部功率计校准):
      real_power = pmic_sum * 1.1451 + 0.5879
    
    注意: PMIC 不监控 5V rail (USB 设备/HAT/NVMe/风扇)，
    所以读数是 SoC 子系统的功率，不是总输入功率。
    """

    # 12 条被监控的分支 (短名 -> 电流key后缀, 电压key后缀)
    BRANCHES = [
        "3V7_WL_SW",   # WiFi/BT
        "3V3_SYS",     # 3.3V 系统
        "1V8_SYS",     # 1.8V 系统
        "DDR_VDD2",    # DDR VDD2
        "DDR_VDDQ",    # DDR VDDQ
        "1V1_SYS",     # 1.1V 系统
        "0V8_SW",      # 0.8V 开关
        "VDD_CORE",    # CPU 核心 (主要功耗来源)
        "3V3_DAC",     # 3.3V DAC
        "3V3_ADC",     # 3.3V ADC
        "0V8_AON",     # 0.8V always-on
        "HDMI",        # HDMI
    ]

    # 线性校正系数
    CORRECTION_SLOPE = 1.1451
    CORRECTION_OFFSET = 0.5879  # Watts

    def __init__(self, apply_correction: bool = True):
        self.apply_correction = apply_correction

    def read_power(self) -> tuple[float, Dict[str, float]]:
        """
        读取 PMIC 数据并计算功率。
        返回: (total_watts, {branch_name: branch_watts})
        """
        result = subprocess.run(
            ["vcgencmd", "pmic_read_adc"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            raise RuntimeError(f"vcgencmd pmic_read_adc 失败: {result.stderr}")

        lines = result.stdout.strip().split('\n')
        currents = {}  # label -> amps
        voltages = {}  # label -> volts

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 格式: "LABEL_A current(N)=X.XXXXXXA" 或 "LABEL_V volt(N)=X.XXXXXXV"
            match = re.match(r'(\S+)\s+(?:current|volt)\(\d+\)=([0-9.]+)[AV]', line)
            if match:
                label = match.group(1)
                value = float(match.group(2))
                # 去掉末尾的 _A 或 _V，得到分支短名
                if label.endswith('_A'):
                    short = label[:-2]
                    currents[short] = value
                elif label.endswith('_V'):
                    short = label[:-2]
                    voltages[short] = value

        # 计算每个分支的功率
        branch_power = {}
        total = 0.0
        for branch in self.BRANCHES:
            i = currents.get(branch, 0.0)
            v = voltages.get(branch, 0.0)
            p = i * v
            branch_power[branch] = p
            total += p

        # 应用线性校正
        if self.apply_correction:
            corrected = total * self.CORRECTION_SLOPE + self.CORRECTION_OFFSET
        else:
            corrected = total

        return corrected, branch_power


class Pi4BEstimator:
    """
    Pi 4B 软件功率估算器 (无外部硬件)
    ------------------------------------
    Pi 4B 的 PMIC (MxL7704) 不暴露电流读数，只有电压。
    因此只能基于 CPU 负载和频率做粗略估算。

    估算模型 (基于公开测试数据):
      - 空闲 (桌面 idle):     ~2.7W
      - 空闲 (headless):      ~2.1W
      - 单核满载:              ~3.4W
      - 四核满载 (stress):     ~6.4W
      - 四核满载 + GPU:        ~7.6W

    线性模型: power ≈ base_idle + (full_load - base_idle) * cpu_usage_fraction
    """

    # 默认参数 (headless, 无 USB 外设)
    IDLE_POWER = 2.1     # W
    FULL_LOAD_POWER = 6.4  # W (4核 stress)

    def __init__(self, idle_power: float = None, full_load_power: float = None):
        self.idle_power = idle_power or self.IDLE_POWER
        self.full_load_power = full_load_power or self.FULL_LOAD_POWER

    def estimate_power(self, cpu_percent: float) -> float:
        """
        根据 CPU 使用率估算功率。
        cpu_percent: 0-100 的 CPU 总使用率
        返回: 估算瓦数
        """
        fraction = min(cpu_percent / 100.0, 1.0)
        return self.idle_power + (self.full_load_power - self.idle_power) * fraction


class INA219Reader:
    """
    INA219 硬件读取器 (需要外接 INA219 传感器)
    -----------------------------------------------
    需安装: pip install pi-ina219
    接线: VCC->3.3V, GND->GND, SDA->GPIO2, SCL->GPIO3
    在电源线上串联 INA219 的 VIN+/VIN- 端子。
    """

    def __init__(self, shunt_ohms: float = 0.1, max_expected_amps: float = 3.0,
                 address: int = 0x40):
        self.shunt_ohms = shunt_ohms
        self.max_expected_amps = max_expected_amps
        self.address = address
        self._ina = None
        self._init_sensor()

    def _init_sensor(self):
        try:
            from ina219 import INA219
            self._ina = INA219(self.shunt_ohms, self.max_expected_amps,
                               address=self.address)
            self._ina.configure()
        except ImportError:
            raise ImportError(
                "需要安装 pi-ina219 库: pip install pi-ina219 --break-system-packages"
            )

    def read_power(self) -> tuple[float, float, float]:
        """
        读取 INA219 传感器。
        返回: (power_mw, voltage_v, current_ma)
        """
        from ina219 import DeviceRangeError
        try:
            voltage = self._ina.voltage()
            current = self._ina.current()
            power = self._ina.power()
            return power / 1000.0, voltage, current  # 转换 mW -> W
        except DeviceRangeError:
            return 0.0, 0.0, 0.0


# ============================================================================
# 系统指标采集 (所有型号通用)
# ============================================================================

class SystemMetrics:
    """轻量级系统指标采集，开销极低"""

    _prev_idle = 0
    _prev_total = 0

    @staticmethod
    def get_cpu_temp() -> float:
        """读取 CPU 温度 (°C)"""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return int(f.read().strip()) / 1000.0
        except Exception:
            try:
                r = subprocess.run(
                    ["vcgencmd", "measure_temp"],
                    capture_output=True, text=True, timeout=2
                )
                m = re.search(r'temp=([\d.]+)', r.stdout)
                return float(m.group(1)) if m else 0.0
            except Exception:
                return 0.0

    @staticmethod
    def get_cpu_freq() -> int:
        """读取 CPU 当前频率 (MHz)"""
        try:
            with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r") as f:
                return int(f.read().strip()) // 1000
        except Exception:
            try:
                r = subprocess.run(
                    ["vcgencmd", "measure_clock", "arm"],
                    capture_output=True, text=True, timeout=2
                )
                m = re.search(r'=(\d+)', r.stdout)
                return int(m.group(1)) // 1000000 if m else 0
            except Exception:
                return 0

    @classmethod
    def get_cpu_usage(cls) -> float:
        """
        读取 CPU 总使用率 (%)，通过 /proc/stat 差分计算。
        开销: 仅读一次文件，无 subprocess 调用。
        """
        try:
            with open("/proc/stat", "r") as f:
                line = f.readline()
            parts = line.split()
            # user, nice, system, idle, iowait, irq, softirq, steal
            values = [int(x) for x in parts[1:9]]
            idle = values[3] + values[4]  # idle + iowait
            total = sum(values)

            diff_idle = idle - cls._prev_idle
            diff_total = total - cls._prev_total
            cls._prev_idle = idle
            cls._prev_total = total

            if diff_total == 0:
                return 0.0
            return (1.0 - diff_idle / diff_total) * 100.0
        except Exception:
            return 0.0

    @staticmethod
    def get_throttled() -> str:
        """读取节流状态"""
        try:
            r = subprocess.run(
                ["vcgencmd", "get_throttled"],
                capture_output=True, text=True, timeout=2
            )
            m = re.search(r'throttled=(0x[0-9a-fA-F]+)', r.stdout)
            return m.group(1) if m else "unknown"
        except Exception:
            return "unknown"


# ============================================================================
# 核心监控器
# ============================================================================

class PowerMonitor:
    """
    功率监控器 - 后台线程运行，不影响主进程
    ============================================

    设计要点:
    1. 使用 daemon 线程，主进程退出时自动终止
    2. 采集操作极轻量 (读 sysfs 文件 + 一次 vcgencmd)
    3. 对 Pi 5: vcgencmd pmic_read_adc 耗时约 10-20ms
    4. 对 Pi 4B: 纯文件读取，耗时 < 1ms
    5. 默认 1 秒采样，对 CPU 密集型 llama.cpp 影响 < 0.5%

    用法:
        mon = PowerMonitor(interval=1.0)
        mon.start()
        # ... 你的 llama.cpp 命令 ...
        mon.stop()
        results = mon.summary()
        mon.to_csv("power_log.csv")
    """

    def __init__(
        self,
        interval: float = 1.0,
        model: str = "auto",
        use_ina219: bool = False,
        ina219_shunt_ohms: float = 0.1,
        callback: Optional[Callable[[PowerReading], None]] = None,
        pi4_idle_power: float = None,
        pi4_full_load_power: float = None,
        apply_correction: bool = True,
    ):
        """
        参数:
            interval: 采样间隔 (秒)，建议 >= 0.5
            model: "auto", "pi5", "pi4b" - 强制指定型号
            use_ina219: 是否使用 INA219 传感器 (Pi 4B)
            ina219_shunt_ohms: INA219 分流电阻值 (Ω)
            callback: 每次采样后的回调函数
            pi4_idle_power: Pi 4B 空闲功率校准值 (W)
            pi4_full_load_power: Pi 4B 满载功率校准值 (W)
            apply_correction: Pi 5 是否应用线性校正
        """
        self.interval = max(interval, 0.1)
        self.callback = callback
        self._readings: List[PowerReading] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0
        self._stop_time: float = 0

        # 检测型号
        if model == "auto":
            self.model = detect_pi_model()
        else:
            self.model = model

        # 初始化读取后端
        self._reader = None
        self._method = "unknown"

        if self.model == "pi5":
            self._reader = Pi5PMICReader(apply_correction=apply_correction)
            self._method = "pmic"
        elif self.model in ("pi4b", "pi4"):
            if use_ina219:
                try:
                    self._reader = INA219Reader(shunt_ohms=ina219_shunt_ohms)
                    self._method = "ina219"
                except (ImportError, Exception) as e:
                    print(f"[PowerMonitor] INA219 不可用 ({e}), 回退到软件估算")
                    self._reader = Pi4BEstimator(pi4_idle_power, pi4_full_load_power)
                    self._method = "estimate"
            else:
                self._reader = Pi4BEstimator(pi4_idle_power, pi4_full_load_power)
                self._method = "estimate"
        else:
            # 未知型号也用估算
            self._reader = Pi4BEstimator(pi4_idle_power, pi4_full_load_power)
            self._method = "estimate"

        # 初始化 CPU 使用率的差分基线
        SystemMetrics.get_cpu_usage()

    def _sample_once(self) -> PowerReading:
        """执行一次采样"""
        ts = time.time()
        cpu_temp = SystemMetrics.get_cpu_temp()
        cpu_freq = SystemMetrics.get_cpu_freq()
        cpu_usage = SystemMetrics.get_cpu_usage()
        throttled = SystemMetrics.get_throttled()

        branches = None
        if self._method == "pmic":
            total_power, branches = self._reader.read_power()
        elif self._method == "ina219":
            total_power, _, _ = self._reader.read_power()
        elif self._method == "estimate":
            total_power = self._reader.estimate_power(cpu_usage)
        else:
            total_power = 0.0

        return PowerReading(
            timestamp=ts,
            total_power_w=total_power,
            cpu_temp_c=cpu_temp,
            cpu_freq_mhz=cpu_freq,
            cpu_usage_percent=round(cpu_usage, 1),
            throttled=throttled,
            method=self._method,
            branches=branches,
        )

    def _monitor_loop(self):
        """监控线程主循环"""
        while not self._stop_event.is_set():
            try:
                reading = self._sample_once()
                with self._lock:
                    self._readings.append(reading)
                if self.callback:
                    self.callback(reading)
            except Exception as e:
                print(f"[PowerMonitor] 采样异常: {e}", file=sys.stderr)
            self._stop_event.wait(self.interval)

    def start(self):
        """启动后台监控线程"""
        if self._thread and self._thread.is_alive():
            print("[PowerMonitor] 已在运行中")
            return

        self._stop_event.clear()
        self._readings.clear()
        self._start_time = time.time()

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

        print(f"[PowerMonitor] 启动 | 型号={self.model} | 方法={self._method} | 间隔={self.interval}s")
        if self._method == "estimate":
            print("[PowerMonitor] ⚠ Pi 4B 无内置电流传感器，功率为软件估算值 (误差可能较大)")
            print("[PowerMonitor]   精确测量建议外接 INA219 传感器或 USB 功率计")

    def stop(self) -> List[PowerReading]:
        """停止监控并返回所有读数"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._stop_time = time.time()
        print(f"[PowerMonitor] 停止 | 采集 {len(self._readings)} 个样本")
        with self._lock:
            return list(self._readings)

    def read_once(self) -> PowerReading:
        """单次即时读取 (不启动后台线程)"""
        return self._sample_once()

    @property
    def readings(self) -> List[PowerReading]:
        """获取当前所有读数的副本 (线程安全)"""
        with self._lock:
            return list(self._readings)

    @property
    def latest(self) -> Optional[PowerReading]:
        """获取最新读数"""
        with self._lock:
            return self._readings[-1] if self._readings else None

    def summary(self, print_output: bool = True) -> Optional[PowerSummary]:
        """计算并 (可选) 打印汇总统计"""
        with self._lock:
            readings = list(self._readings)

        if not readings:
            print("[PowerMonitor] 无数据")
            return None

        powers = [r.total_power_w for r in readings]
        temps = [r.cpu_temp_c for r in readings]
        cpus = [r.cpu_usage_percent for r in readings]

        duration = (self._stop_time or time.time()) - self._start_time
        # 用梯形法估算总能耗
        energy_j = 0.0
        for i in range(1, len(readings)):
            dt = readings[i].timestamp - readings[i - 1].timestamp
            avg_p = (readings[i].total_power_w + readings[i - 1].total_power_w) / 2
            energy_j += avg_p * dt

        s = PowerSummary(
            duration_s=round(duration, 2),
            samples=len(readings),
            method=readings[0].method,
            avg_power_w=round(statistics.mean(powers), 3),
            min_power_w=round(min(powers), 3),
            max_power_w=round(max(powers), 3),
            std_power_w=round(statistics.stdev(powers), 3) if len(powers) > 1 else 0.0,
            avg_temp_c=round(statistics.mean(temps), 1),
            max_temp_c=round(max(temps), 1),
            avg_cpu_percent=round(statistics.mean(cpus), 1),
            total_energy_wh=round(energy_j / 3600, 4),
            total_energy_j=round(energy_j, 2),
        )

        if print_output:
            print("\n" + "=" * 60)
            print("  树莓派功耗监控报告")
            print("=" * 60)
            print(f"  型号:         {self.model}")
            print(f"  测量方法:     {s.method}")
            if s.method == "estimate":
                print(f"                (⚠ 软件估算，非实测)")
            print(f"  监控时长:     {s.duration_s:.1f} 秒")
            print(f"  采样数:       {s.samples}")
            print(f"  ─────────────────────────────────────")
            print(f"  平均功率:     {s.avg_power_w:.3f} W")
            print(f"  最小功率:     {s.min_power_w:.3f} W")
            print(f"  最大功率:     {s.max_power_w:.3f} W")
            print(f"  标准差:       {s.std_power_w:.3f} W")
            print(f"  ─────────────────────────────────────")
            print(f"  平均温度:     {s.avg_temp_c:.1f} °C")
            print(f"  最高温度:     {s.max_temp_c:.1f} °C")
            print(f"  平均CPU使用:  {s.avg_cpu_percent:.1f} %")
            print(f"  ─────────────────────────────────────")
            print(f"  总能耗:       {s.total_energy_j:.2f} J ({s.total_energy_wh:.4f} Wh)")
            print("=" * 60 + "\n")

        return s

    def to_csv(self, filepath: str):
        """将所有读数导出为 CSV"""
        with self._lock:
            readings = list(self._readings)

        if not readings:
            print("[PowerMonitor] 无数据可导出")
            return

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            # 表头
            header = [
                "timestamp", "datetime", "power_w", "cpu_temp_c",
                "cpu_freq_mhz", "cpu_usage_pct", "throttled", "method"
            ]
            # Pi 5 增加分支功率列
            if readings[0].branches:
                header.extend([f"{b}_W" for b in Pi5PMICReader.BRANCHES])
            writer.writerow(header)

            for r in readings:
                row = [
                    f"{r.timestamp:.3f}",
                    datetime.fromtimestamp(r.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    f"{r.total_power_w:.4f}",
                    f"{r.cpu_temp_c:.1f}",
                    r.cpu_freq_mhz,
                    f"{r.cpu_usage_percent:.1f}",
                    r.throttled,
                    r.method,
                ]
                if r.branches:
                    row.extend([f"{r.branches.get(b, 0):.6f}" for b in Pi5PMICReader.BRANCHES])
                writer.writerow(row)

        print(f"[PowerMonitor] CSV 已保存: {filepath}")


# ============================================================================
# 便捷函数 (一行调用)
# ============================================================================

def monitor_command(
    command: str,
    interval: float = 1.0,
    csv_path: Optional[str] = None,
    **kwargs
) -> PowerSummary:
    """
    监控一个 shell 命令的功耗。

    示例:
        summary = monitor_command(
            "./llama-cli -m model.gguf -p 'Hello world' -n 128",
            interval=0.5,
            csv_path="llama_power.csv"
        )

    参数:
        command: 要执行的 shell 命令
        interval: 采样间隔 (秒)
        csv_path: 可选的 CSV 输出路径
        **kwargs: 传递给 PowerMonitor 的额外参数

    返回: PowerSummary
    """
    mon = PowerMonitor(interval=interval, **kwargs)
    mon.start()

    try:
        process = subprocess.run(command, shell=True)
        exit_code = process.returncode
    except KeyboardInterrupt:
        exit_code = -1
        print("\n[monitor_command] 被用户中断")

    readings = mon.stop()
    summary = mon.summary()

    if csv_path:
        mon.to_csv(csv_path)

    print(f"[monitor_command] 命令退出码: {exit_code}")
    return summary


def quick_read() -> PowerReading:
    """快速单次读取当前功率"""
    mon = PowerMonitor()
    return mon.read_once()


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="树莓派功耗监控工具 (支持 Pi 5 PMIC / Pi 4B 估算 / INA219)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 独立运行，每 0.5 秒采样，输出 CSV
  python3 rpi_power_monitor.py --interval 0.5 --output power.csv

  # 包裹 llama.cpp 命令
  python3 rpi_power_monitor.py --wrap "./llama-cli -m model.gguf -p 'Hi' -n 128"

  # 使用 INA219 硬件传感器 (Pi 4B)
  python3 rpi_power_monitor.py --ina219 --interval 1.0

  # 快速读一次
  python3 rpi_power_monitor.py --once
        """
    )
    parser.add_argument("--interval", "-i", type=float, default=1.0,
                        help="采样间隔秒数 (默认: 1.0)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="CSV 输出文件路径")
    parser.add_argument("--wrap", "-w", type=str, default=None,
                        help="要监控的 shell 命令")
    parser.add_argument("--ina219", action="store_true",
                        help="使用 INA219 传感器 (需要硬件)")
    parser.add_argument("--once", action="store_true",
                        help="单次读取后退出")
    parser.add_argument("--model", type=str, default="auto",
                        choices=["auto", "pi5", "pi4b"],
                        help="强制指定型号 (默认: auto)")
    parser.add_argument("--no-correction", action="store_true",
                        help="Pi 5: 不应用线性校正")
    parser.add_argument("--live", action="store_true",
                        help="实时打印每次采样结果")

    args = parser.parse_args()

    # 单次读取
    if args.once:
        mon = PowerMonitor(model=args.model, use_ina219=args.ina219,
                           apply_correction=not args.no_correction)
        r = mon.read_once()
        print(f"功率: {r.total_power_w:.3f} W | "
              f"温度: {r.cpu_temp_c:.1f}°C | "
              f"频率: {r.cpu_freq_mhz} MHz | "
              f"CPU: {r.cpu_usage_percent:.1f}% | "
              f"方法: {r.method}")
        return

    # 实时打印的回调
    live_callback = None
    if args.live or (not args.wrap):
        def live_callback(r: PowerReading):
            ts = datetime.fromtimestamp(r.timestamp).strftime("%H:%M:%S")
            print(f"[{ts}] {r.total_power_w:6.3f}W | "
                  f"{r.cpu_temp_c:5.1f}°C | "
                  f"{r.cpu_freq_mhz:4d}MHz | "
                  f"CPU {r.cpu_usage_percent:5.1f}% | "
                  f"{r.method}")

    # 包裹命令模式
    if args.wrap:
        summary = monitor_command(
            args.wrap,
            interval=args.interval,
            csv_path=args.output,
            model=args.model,
            use_ina219=args.ina219,
            apply_correction=not args.no_correction,
            callback=live_callback if args.live else None,
        )
        return

    # 独立持续监控模式 (Ctrl+C 停止)
    mon = PowerMonitor(
        interval=args.interval,
        model=args.model,
        use_ina219=args.ina219,
        apply_correction=not args.no_correction,
        callback=live_callback,
    )

    def signal_handler(sig, frame):
        print("\n[Ctrl+C] 正在停止...")
        mon.stop()
        mon.summary()
        if args.output:
            mon.to_csv(args.output)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    mon.start()
    print("按 Ctrl+C 停止监控...\n")

    # 保持主线程存活
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
