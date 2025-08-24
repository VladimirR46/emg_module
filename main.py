import os, re, json, asyncio
from typing import Dict, List, Optional
from mne_lsl.stream import StreamLSL
from pylsl import resolve_streams
import logging
import imufusion
import numpy as np
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)04d %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class AsyncTCP:
    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self.writer: Optional[asyncio.StreamWriter] = None

    async def _ensure_connected(self):
        if self.writer is None or self.writer.is_closing():
            try:
                _, self.writer = await asyncio.open_connection(self.host, self.port)
            except Exception:
                self.writer = None

    async def send_json(self, obj: dict):
        await self._ensure_connected()
        if self.writer is None:
            return
        try:
            self.writer.write((json.dumps(obj) + "\n").encode("utf-8"))
            await self.writer.drain()
        except Exception:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
            self.writer = None  # повторим попытку на следующей отправке

class Sensor:
    def __init__(self, serial: str, name: str, source_id: str):
        self.serial = serial
        self.stream = StreamLSL(bufsize=10.0, name=name, stype="Data", source_id=source_id)
        self.stream.connect(acquisition_delay=0.01, processing_flags="all", timeout=4.0)
        #self.stream.filter(l_freq=None, h_freq=40.0)
        self.above = False
        self.tcp = None
        self.prev_ts: Optional[float] = None

        self.sfreq = float(self.stream.info["sfreq"]) if float(self.stream.info["sfreq"]) > 0 else 100.0
        self.dt = 1.0 / self.sfreq

        self.ahrs = imufusion.Ahrs()
        #self.offset = imufusion.Offset(int(self.sfreq))
        self.ahrs.settings = imufusion.Settings(imufusion.CONVENTION_NED,  0.5, 100,  10,  10, int(2 * int(self.sfreq)))

        self.last_print = 0.0
        self.print_period = 1.0 / 50.0

        self.initA = None
        self.initG = None

        self.angle = 0.0

    def _update_ahrs_chunk(self, data: np.ndarray, ts: np.ndarray):
        acc = data[[1, 2, 3], :].T * 1e-3
        gyr = data[[4, 5, 6], :].T * 1e-3
        prev = self.prev_ts
        for t, a, g in zip(ts.astype(float), acc, gyr):
            if prev is None:
                prev = t
                continue
            dt = t - prev
            prev = t
            self.ahrs.update_no_magnetometer(g, a, dt)
        self.prev_ts = prev

    async def data_process(self, data, ts):
        y = data[1]
        vmax = float(y.max())
        if vmax > 700:
            if not self.above:
                await self.tcp.send_json({"serial": int(self.serial), "acc_trigger": 1})
                self.above = True
        else:
            self.above = False

        self._update_ahrs_chunk(data, ts)
        roll, pitch, yaw = self.ahrs.quaternion.to_euler()

        if abs(self.angle-roll) > 3.0:
            await self.tcp.send_json({"serial": int(self.serial), "angle": int(roll)})
            self.angle = roll

        now = time.perf_counter()
        if now - self.last_print >= self.print_period:
            print(f"[{self.serial}] yaw={yaw:7.2f}°, pitch={pitch:7.2f}°, roll={roll:7.2f}°", end="\r")
            self.last_print = now

    async def run(self, tcp: AsyncTCP):
        self.tcp = tcp
        info = self.stream.info
        logger.info(f"[{self.serial}] connected: name={self.stream.name} sfreq={info['sfreq']}Hz chans={len(info['ch_names'])}")
        try:
            while True:
                n = self.stream.n_new_samples
                if n:
                    winsize = n / int(self.stream.info["sfreq"])
                    data, ts = self.stream.get_data(winsize=winsize)  # data:(n_ch,n), ts:(n,)
                    await self.data_process(data, ts)
                await asyncio.sleep(0)
        except Exception as e:
            logger.error(f"Error in main processing loop: {e}")
        finally:
            self.stream.disconnect()
            logger.info("Shutdown complete")

def discover_sensors() -> List[Sensor]:
    streams = resolve_streams(2.0)
    sensors: Dict[str, Sensor] = {}
    for s in streams:
        try:
            m = re.fullmatch(r"EMGsens-(\d+)-Data", s.name)
            if not m:
                continue
            serial = m.group(1)
            if serial not in sensors:
                sensors[serial] = Sensor(serial, s.name, s.source_id)
        except Exception as e:
            print(f"Error: descovery {e}")
            continue
    return list(sensors.values())

async def main():
    tcp = AsyncTCP("127.0.0.1", 8080)
    sensors = discover_sensors()
    if not sensors:
        print("No EMGsens-*-Data streams found.")
        return
    print("Sensors:", [s.serial for s in sensors])

    tasks = [asyncio.create_task(s.run(tcp)) for s in sensors]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Main process received keyboard interrupt")