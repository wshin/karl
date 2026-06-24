"""Karl's animated face — a standalone window driven by stdin state commands.

main.py runs this as a subprocess when `karl --face` is used. It reads one command
per line from stdin: idle | listening | thinking | speaking | quit, and animates a
simple face accordingly (blinking, eyes that glance around while listening, a mouth
that moves while speaking, thinking dots).

Pygame must run on this process's main thread (a macOS requirement); a daemon thread
reads stdin. If pygame or a display is unavailable, it exits quietly so Karl keeps
working without a face.
"""
import math
import random
import sys
import threading


def main() -> None:
    try:
        import pygame
    except Exception:
        return

    state = {"v": "idle"}

    def reader():
        try:
            for line in sys.stdin:
                cmd = line.strip().lower()
                if cmd == "quit":
                    state["v"] = "quit"
                    return
                if cmd in ("idle", "listening", "thinking", "speaking"):
                    state["v"] = cmd
        except Exception:
            pass

    threading.Thread(target=reader, daemon=True).start()

    try:
        pygame.init()
        W, H = 420, 470
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Karl")
        font = pygame.font.SysFont("Helvetica", 22)
    except Exception:
        return

    clock = pygame.time.Clock()
    BG, SKIN, EYE, PUP, MOUTH = (16, 18, 26), (58, 68, 96), (236, 239, 246), (26, 29, 38), (15, 16, 23)
    RINGS = {"idle": (74, 84, 110), "listening": (88, 200, 128),
             "thinking": (232, 190, 92), "speaking": (92, 178, 232)}
    LABELS = {"idle": "", "listening": "listening…", "thinking": "thinking…", "speaking": "speaking"}

    cx, cy = W // 2, H // 2 - 16
    t = blink = 0.0
    next_blink = random.uniform(2.0, 4.0)
    mouth = 0.05
    running = True
    while running:
        dt = clock.tick(30) / 1000.0
        t += dt
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
        s = state["v"]
        if s == "quit":
            break

        next_blink -= dt
        if next_blink <= 0:
            blink, next_blink = 0.16, random.uniform(2.5, 5.5)
        blink = max(0.0, blink - dt)
        eye_open = 0.12 if blink > 0 else 1.0

        if s == "speaking":
            target = 0.30 + 0.55 * abs(math.sin(t * 10.5)) * (0.55 + 0.45 * random.random())
        elif s == "thinking":
            target = 0.10
        else:
            target = 0.05
        mouth += (target - mouth) * min(1.0, dt * 20)

        screen.fill(BG)
        ring = RINGS.get(s, RINGS["idle"])
        pygame.draw.circle(screen, SKIN, (cx, cy), 150)
        pygame.draw.circle(screen, ring, (cx, cy), 150, 6)

        ex, ey, er = 56, -34, 26
        for sign in (-1, 1):
            look = math.sin(t * 1.4) * 7 if s == "listening" else 0
            eh = max(2, int(er * eye_open))
            pygame.draw.ellipse(screen, EYE, (cx + sign * ex - er, cy + ey - eh, er * 2, eh * 2))
            if eye_open > 0.5:
                pygame.draw.circle(screen, PUP, (int(cx + sign * ex + look), cy + ey), 10)

        if s == "thinking":
            for i in range(3):
                a = (math.sin(t * 4 - i * 0.7) + 1) / 2
                pygame.draw.circle(screen, RINGS["thinking"], (cx - 26 + i * 26, cy - 178), 5 + int(3 * a))

        mh = int(8 + mouth * 60)
        pygame.draw.ellipse(screen, MOUTH, (cx - 46, cy + 58 - mh // 2, 92, mh))

        lbl = LABELS.get(s, "")
        if lbl:
            surf = font.render(lbl, True, ring)
            screen.blit(surf, (cx - surf.get_width() // 2, H - 40))

        pygame.display.flip()

    try:
        pygame.quit()
    except Exception:
        pass


if __name__ == "__main__":
    main()
