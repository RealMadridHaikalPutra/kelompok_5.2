"""Koreksi berbasis posisi + voting antar-bacaan untuk plat Indonesia.

Format: [1-2 huruf wilayah][1-4 angka][0-3 huruf seri]
  * di segmen huruf, angka yang mirip huruf diubah jadi huruf (0->O, 8->B ...)
  * di segmen angka, huruf yang mirip angka diubah jadi angka (O->0, B->8 ...)
  * beberapa bacaan (fast-alpr + varian EasyOCR) divoting per karakter,
    dibobot dengan keyakinan.
"""

from __future__ import annotations

TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "J": "1",
            "Z": "2", "S": "5", "G": "6", "T": "7", "B": "8", "A": "4"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G",
             "7": "T", "8": "B"}

# Huruf yang SUDAH benar di segmen huruf tidak boleh diubah. Khusus '0' di
# segmen huruf: bila tidak ada bacaan lain, jadikan 'O' (voting bisa
# menggantinya menjadi Q/D bila ada bacaan yang menyebut demikian).
BAD = 99


def koreksi(teks: str) -> tuple[str, int]:
    """Pilih pemisahan (wilayah|angka|seri) dengan koreksi paling sedikit.

    Mengembalikan (teks_terkoreksi, biaya). biaya = jumlah karakter yang diubah.
    """
    n = len(teks)
    best: tuple[str, int] | None = None
    for a in (1, 2):
        for b in range(1, 5):
            c = n - a - b
            if not 0 <= c <= 3:
                continue
            pre, dig, suf = teks[:a], teks[a:a + b], teks[a + b:]
            cost, out = 0, []
            for ch in pre:
                if ch.isdigit():
                    cost += 1
                    ch = TO_LETTER.get(ch, ch)
                    if ch.isdigit():
                        cost += 3
                out.append(ch)
            for ch in dig:
                if ch.isalpha():
                    cost += 1
                    ch = TO_DIGIT.get(ch, ch)
                    if ch.isalpha():
                        cost += 3
                out.append(ch)
            for ch in suf:
                if ch.isdigit():
                    cost += 1
                    ch = TO_LETTER.get(ch, ch)
                    if ch.isdigit():
                        cost += 3
                out.append(ch)
            cand = ("".join(out), cost)
            if best is None or cand[1] < best[1]:
                best = cand
    return best if best else (teks, BAD)


def vote(reads: list[tuple[str, float, str]]):
    """reads = [(teks, conf, nama)]. Kembalikan (teks, kesepakatan 0..1) / None."""
    items: list[tuple[str, float]] = []
    for teks, conf, _ in reads:
        fixed, cost = koreksi(teks)
        if cost >= BAD or cost > 3:
            continue
        items.append((fixed, max(conf, 0.05) * (0.85 ** cost)))
    if not items:
        return None

    by_len: dict[int, float] = {}
    for k, w in items:
        by_len[len(k)] = by_len.get(len(k), 0.0) + w
    length = max(by_len, key=by_len.get)
    pool = [(k, w) for k, w in items if len(k) == length]

    out, agree = [], []
    for i in range(length):
        tally: dict[str, float] = {}
        for k, w in pool:
            tally[k[i]] = tally.get(k[i], 0.0) + w
        top = max(tally, key=tally.get)
        out.append(top)
        agree.append(tally[top] / sum(tally.values()))
    return "".join(out), sum(agree) / len(agree)


def sepakat(reads: list[tuple[str, float, str]], minimal: int = 2) -> bool:
    """True bila >= `minimal` bacaan menghasilkan teks terkoreksi yang SAMA."""
    seen: dict[str, int] = {}
    for teks, _, _ in reads:
        fixed, cost = koreksi(teks)
        if cost <= 1:
            seen[fixed] = seen.get(fixed, 0) + 1
    return any(v >= minimal for v in seen.values())