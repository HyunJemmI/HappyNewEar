from pathlib import Path
import argparse
import json
import logging
import socket
import subprocess
import threading
import time

import pygame

from Classification import SoundClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PORT = 9001
DEST_PORT = 9000
HOST = "127.0.0.1"
DISPLAY_SIZE = (800, 480)
PLOT_CENTER = (335, 240)
PLOT_SCALE = 220
RAW_WINDOW_SECONDS = 1.0
SOCKET_EMPTY_LIMIT = 10
RAW_RECV_SIZE = 8192
DEST_RECV_SIZE = 4096

DANGER_SOUNDS = {
    "경적": 3,
    "고함": 1,
    "폭발": 3,
    "충돌": 3,
    "화재경보": 3,
    "알람": 1,
    "응급차량": 2,
    "사이렌": 2,
}

CHANNEL_COLORS = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 0),
]


class HappyNewEarApp:
    def __init__(self, args):
        self.args = args
        self.classifier = SoundClassifier(args.model_path, args.class_map_path)
        self.positions = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.results = ["", "", "", ""]
        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.alert_lock = threading.Lock()
        self.alert_running = False
        self.error_count = 0
        self.odas_process = None

    def run(self):
        raw_server = self.create_server(RAW_PORT)
        dest_server = self.create_server(DEST_PORT)

        threads = [
            threading.Thread(target=self.receive_raw_audio, args=(raw_server,), daemon=True),
            threading.Thread(target=self.receive_position_data, args=(dest_server,), daemon=True),
            threading.Thread(target=self.run_display, daemon=True),
        ]

        for thread in threads:
            thread.start()

        try:
            self.start_odas()
            while not self.stop_event.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.stop_event.set()
        finally:
            self.stop_event.set()
            self.stop_odas()
            raw_server.close()
            dest_server.close()

            for thread in threads:
                thread.join(timeout=1)

    @staticmethod
    def create_server(port):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, port))
        server.listen(1)
        logging.info("Listening on %s:%s", HOST, port)
        return server

    def start_odas(self):
        odas_bin = Path(self.args.odas_bin)
        odas_config = Path(self.args.odas_config)
        if not odas_bin.exists():
            logging.warning("ODAS binary not found: %s", odas_bin)
            return

        command = [str(odas_bin), "-c", str(odas_config)]
        self.odas_process = subprocess.Popen(command, cwd=str(odas_bin.parent))
        logging.info("ODAS started: %s", " ".join(command))

    def stop_odas(self):
        if self.odas_process and self.odas_process.poll() is None:
            self.odas_process.terminate()
            try:
                self.odas_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.odas_process.kill()

    def receive_raw_audio(self, server):
        logging.info("Waiting for ODAS raw audio stream")
        client, address = server.accept()
        logging.info("Raw audio client connected: %s", address)

        raw_buffers = []
        window_started_at = time.time()
        empty_count = 0

        with client:
            while not self.stop_event.is_set():
                buffer = client.recv(RAW_RECV_SIZE)
                if not buffer:
                    empty_count += 1
                    if empty_count > SOCKET_EMPTY_LIMIT:
                        break
                    continue

                empty_count = 0
                raw_buffers.append(buffer)

                if time.time() - window_started_at >= RAW_WINDOW_SECONDS:
                    data = raw_buffers
                    threading.Thread(target=self.classify_audio_window, args=(data,), daemon=True).start()
                    raw_buffers = []
                    window_started_at = time.time()

    def classify_audio_window(self, buffers):
        data = self.classifier.tonumpy(buffers)
        data = self.classifier.preprocess(data)
        if data.size == 0:
            return

        channel_results = []
        for channel_index in range(4):
            channel_data = data[channel_index::4]
            channel_results.append(self.classifier.classifier(channel_data))

        with self.state_lock:
            self.results = channel_results

    def receive_position_data(self, server):
        logging.info("Waiting for ODAS position stream")
        client, address = server.accept()
        logging.info("Position client connected: %s", address)

        empty_count = 0
        with client:
            while not self.stop_event.is_set():
                buffer = client.recv(DEST_RECV_SIZE)
                if not buffer:
                    empty_count += 1
                    if empty_count > SOCKET_EMPTY_LIMIT:
                        break
                    continue

                empty_count = 0
                try:
                    payload = json.loads(buffer.decode())
                    self.update_positions(payload)
                except Exception as exc:
                    self.error_count += 1
                    logging.debug("Position parse error: %s", exc)

    def update_positions(self, payload):
        positions = [[0, 0], [0, 0], [0, 0], [0, 0]]
        for channel_index, source in enumerate(payload.get("src", [])[:4]):
            positions[channel_index][0] = int(source["x"] * PLOT_SCALE + PLOT_CENTER[0])
            positions[channel_index][1] = int(source["y"] * PLOT_SCALE + PLOT_CENTER[1])

        with self.state_lock:
            self.positions = positions

    def run_display(self):
        pygame.init()
        screen = pygame.display.set_mode(DISPLAY_SIZE, pygame.FULLSCREEN)
        pygame.display.set_caption("HappyNewEar")
        font = pygame.font.SysFont("nanumgothic", 15)
        clock = pygame.time.Clock()

        try:
            while not self.stop_event.is_set():
                clock.tick(30)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.stop_event.set()

                with self.state_lock:
                    positions = [position[:] for position in self.positions]
                    results = list(self.results)

                self.draw_screen(screen, font, positions, results)
                pygame.display.flip()
        finally:
            pygame.quit()

    def draw_screen(self, screen, font, positions, results):
        black = (0, 0, 0)
        gray = (125, 125, 125)
        white = (255, 255, 255)

        screen.fill(white)
        self.draw_danger_image(screen, results)

        pygame.draw.line(screen, black, (335, 10), (335, 470), 3)
        pygame.draw.line(screen, black, (10, 240), (660, 240), 3)
        pygame.draw.rect(screen, gray, [10, 10, 650, 460], 5)

        for index, position in enumerate(positions):
            pygame.draw.circle(screen, CHANNEL_COLORS[index], position, 4)
            label = font.render(results[index], True, CHANNEL_COLORS[index])
            screen.blit(label, (670, 95 + index * 40))

    def draw_danger_image(self, screen, results):
        for result in results:
            if result not in DANGER_SOUNDS:
                continue

            image_path = self.args.image_dir / f"{result}.png"
            if image_path.exists():
                image = pygame.image.load(str(image_path))
                image = pygame.transform.scale(image, (460, 460))
                screen.blit(image, (105, 10))

            self.start_alert(DANGER_SOUNDS[result])
            break

    def start_alert(self, rate):
        with self.alert_lock:
            if self.alert_running:
                return
            self.alert_running = True

        threading.Thread(target=self.play_alert, args=(rate,), daemon=True).start()

    def play_alert(self, rate):
        try:
            for _ in range(rate):
                subprocess.run(
                    ["play", "-nq", "-t", "alsa", "synth", "0.5", "sine", "100"],
                    check=False,
                )
        finally:
            with self.alert_lock:
                self.alert_running = False


def parse_args():
    parser = argparse.ArgumentParser(description="Run HappyNewEar sound danger monitor.")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "csv" / "yamnet.tflite")
    parser.add_argument("--class-map-path", type=Path, default=PROJECT_ROOT / "csv" / "yamnet_class_map.csv")
    parser.add_argument("--image-dir", type=Path, default=PROJECT_ROOT / "img")
    parser.add_argument("--odas-bin", type=Path, default=PROJECT_ROOT / "odas" / "bin" / "odaslive")
    parser.add_argument("--odas-config", type=Path, default=PROJECT_ROOT / "odas" / "bin" / "odas.cfg")
    parser.add_argument("--log-path", type=Path, default=PROJECT_ROOT / "logs" / "happynewear.log")
    return parser.parse_args()


def setup_logging(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main():
    args = parse_args()
    setup_logging(args.log_path)
    app = HappyNewEarApp(args)
    app.run()
    logging.info("HappyNewEar stopped. position parse errors=%s", app.error_count)


if __name__ == "__main__":
    main()
