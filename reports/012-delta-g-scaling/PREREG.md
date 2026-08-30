# Pre-rejestracja: siatka skalowania (δ, g) — kandydat na raport 012

Zarejestrowano 2026-08-30 ~13:00 CEST, PRZED biegiem (start 15:00).
Motywacja: model autora zakłada hierarchię g ~ 10^10, δ ~ 10^-10;
wszystkie dotychczasowe pomiary siedzą przy (0.3–0.125, 8). Zmierzony
flip znaku λ_axis między δ=0.3 a 1/8 (raport 011) dowodzi, że część
obserwabli jest w reżimie konkurencji członów — ekstrapolacja przez
9 rzędów wymaga zmierzonych wykładników, nie nadziei.

## Siatka
δ ∈ {1/8, 1/64, 1/512} × g ∈ {8, 64, 512} — pełne 9 punktów
(kroki ×8 w logu; przekątna δ=1/g to oś hierarchii autora).
Punkt (1/8, 8) pokrywa się z dotychczasową konfiguracją elegancką.

## Zmiana definicji roboczej (jawna): potencjał WZGLĘDNY
Dotychczas V4 = Σ_p (tr((Mη)^p) − C_p)², C_p = g^p + 1 + δ^p.
Przy g=512: C_4 ≈ 6.9e10, kwadraty ~1e21 → uwarunkowanie ~g^8 i
ryzyko kancelacji w double. Bieg używa formy bezwymiarowej
V4_rel = Σ_p (tr((Mη)^p)/C_p − 1)², W1 bez zmian (arbitralne).
Przy (0.3, 8) obie formy różnią się tylko przeskalowaniem wag per-p;
kontrola spójności: punkt (1/8, 8) porównany z pomiarami sondy δ.

## Obserwable (na working-jeżu 32³, po wspólnym protokole
Adam 1000 + 4×L-BFGS z E_levels; wszystko per punkt siatki)
1. sign_eta, sign_G: znak netto czasowego wkładu I₁ w kontrakcji
   η i G (frozen boost tangent) — PRZEWIDYWANIE: znaki stałe na całej
   siatce (własność strukturalna projektora, nie skali).
2. C1, C2, ω_pred = √(C1/C2) — PRZEWIDYWANIE: ω_pred zdominowane
   przez oś radialną (wartość 1): zależność od δ słaba (∝δ² wkłady
   transwersalne), od g przez profil rdzenia — wykładnik do zmierzenia.
3. Mini-drabina (0, ω_pred, 1.5·ω_pred), γ z budżetu 5%:
   interior TAK/NIE + głębokość — PRZEWIDYWANIE: interior wszędzie
   (mechanizm δ/g-odporny); głębokość skaluje z C1²/C2.
4. Krzywizna modu mieszania osi 3-4 — PRZEWIDYWANIE: gradientowo
   zdominowana (sonda δ), słaba zależność od δ; od g nieznana.
5. I_pure, I_comb (inercje rotacyjne, stały L=48) — PRZEWIDYWANIE:
   wkład pary (1,δ): →const przy δ→0; wkład (δ,0): ∝δ².
6. E_stat, ‖g‖∞, trajektoria E_levels (kontrola zbieżności).
7. Diagnostyka precyzji: |tr/C_p−1| zakresy, porównanie energii
   float64 vs float32 (względne), max ulp-strata w V4_rel.

## Kryteria
- Fit log-log per obserwabla: osobno wzdłuż wierszy (δ przy stałym g),
  kolumn (g przy stałym δ) i przekątnej. Obserwabla "przenośna do
  hierarchii autora" = wykładniki spójne z A≠0 (plateau) albo czysty
  ∝δ^n / g^-n z n≥1 zmierzonym na ≥2 oktawach.
- Flip znaku któregokolwiek z (1) unieważnia ekstrapolację tej
  obserwabli i jest wynikiem samym w sobie.
- Punkty z diagnostyką precyzji poniżej progu (float32-różnica >1e-3
  względnie) są flagowane i wyłączane z fitów.

## Protokół wykonania
GPU0 (CUDA_VISIBLE_DEVICES=0), start 15:00 (automatyczna kolejka),
kolejność punktów: rosnące g (najtrudniejsze na końcu), w ramach g
malejące δ. Wyniki: results_grid.json + pola per punkt (npz).
