import json
import math
import random
import statistics

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dp / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    )
    return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def solve_linear_system_3x3(A, b):
    """Menyelesaikan persamaan matriks A * beta = b ukuran 3x3 dengan Eliminasi Gauss-Jordan

    untuk estimasi parameter Weighted Least Squares (WLS) pada model GWR.
    """
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    n = 3
    for i in range(n):
        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            pivot = 1e-12
        for j in range(i, n + 1):
            M[i][j] /= pivot
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(i, n + 1):
                    M[k][j] -= factor * M[i][j]
    return [M[i][n] for i in range(n)]

def generate_spatial_assets_and_morans_i(n_assets=30):
    print("=" * 85)
    print("TAHAP 1: UJI AUTOKORELASI SPASIAL GLOBAL MORAN'S I PADA KERUSAKAN ASET PUBLIK")
    print("=" * 85)

    random.seed(2026)
    assets = []
    for i in range(1, n_assets + 1):
        if i <= n_assets // 2:
            lat = random.gauss(-3.7920, 0.008)
            lon = random.gauss(102.2520, 0.008)
            age_years = random.uniform(6.0, 14.0)
            daily_load = random.uniform(50.0, 95.0)
            local_beta_age = 4.2
        else:
            lat = random.gauss(-3.8250, 0.008)
            lon = random.gauss(102.2950, 0.008)
            age_years = random.uniform(1.0, 7.0)
            daily_load = random.uniform(20.0, 55.0)
            local_beta_age = 1.8

        damage_score = (
            12.0
            + (local_beta_age * age_years)
            + (0.35 * daily_load)
            + random.gauss(0, 3.0)
        )
        damage_score = min(max(damage_score, 5.0), 100.0)

        assets.append({
            "asset_id": f"ASET-{i:03d}",
            "lat": lat,
            "lon": lon,
            "age_years": age_years,
            "utilization_rate": daily_load,
            "damage_index": damage_score,
        })

    N = len(assets)
    y = [a["damage_index"] for a in assets]
    y_mean = statistics.mean(y)
    y_dev = [val - y_mean for val in y]

    W_sum = 0.0
    numerator = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                dist = haversine_km(
                    assets[i]["lat"],
                    assets[i]["lon"],
                    assets[j]["lat"],
                    assets[j]["lon"],
                )
                w_ij = 1.0 / max(dist, 0.05) if dist <= 4.0 else 0.0
                W_sum += w_ij
                numerator += w_ij * y_dev[i] * y_dev[j]

    denom = sum(d**2 for d in y_dev)
    moran_I = (N / W_sum) * (numerator / denom)

    expected_I = -1.0 / (N - 1)
    var_I = 0.0185 
    z_score = (moran_I - expected_I) / math.sqrt(var_I)
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))

    print(f"Jumlah Titik Aset Publik (N)   : {N} lokasi")
    print(f"Indeks Global Moran's I        : {moran_I:.4f} (Ekspektasi E[I] = {expected_I:.4f})")
    print(f"Z-Score Statistik              : {z_score:.4f} (p-value = {p_val:.6e})")
    pola = (
        "CLUSTERED (Mengelompok Secara Spasial Signifikan)"
        if (moran_I > expected_I and p_val < 0.05)
        else "ACAK (Random)"
    )
    print(f"Interpretasi Pola Spasial      : {pola}\n")
    return assets

def run_geographically_weighted_regression(assets, bandwidth_km=2.5):
    print("=" * 85)
    print(f"TAHAP 2: GEOGRAPHICALLY WEIGHTED REGRESSION (GWR) - BANDWIDTH = {bandwidth_km} KM")
    print("=" * 85)

    N = len(assets)
    rss_gwr = 0.0
    y_all = [a["damage_index"] for a in assets]
    y_mean = statistics.mean(y_all)
    tss = sum((y - y_mean) ** 2 for y in y_all)

    for i in range(N):
        XTX = [[0.0, 0.0, 0.0] for _ in range(3)]
        XTY = [0.0, 0.0, 0.0]

        for j in range(N):
            dist = haversine_km(
                assets[i]["lat"],
                assets[i]["lon"],
                assets[j]["lat"],
                assets[j]["lon"],
            )
            w_ij = math.exp(-0.5 * ((dist / bandwidth_km) ** 2))

            x_j = [1.0, assets[j]["age_years"], assets[j]["utilization_rate"]]
            y_j = assets[j]["damage_index"]

            for r in range(3):
                XTY[r] += w_ij * x_j[r] * y_j
                for c in range(3):
                    XTX[r][c] += w_ij * x_j[r] * x_j[c]

        beta_local = solve_linear_system_3x3(XTX, XTY)
        assets[i]["gwr_b0_intercept"] = beta_local[0]
        assets[i]["gwr_b1_age"] = beta_local[1]
        assets[i]["gwr_b2_load"] = beta_local[2]

        y_hat = (
            beta_local[0]
            + beta_local[1] * assets[i]["age_years"]
            + beta_local[2] * assets[i]["utilization_rate"]
        )
        assets[i]["gwr_predicted"] = y_hat
        rss_gwr += (assets[i]["damage_index"] - y_hat) ** 2

    r2_gwr = 1.0 - (rss_gwr / tss)
    print(f"Goodness-of-Fit Model Spasial GWR (R-Squared) : {r2_gwr:.4f} ({r2_gwr*100:.2f}%)\n")
    print("Sampel Variasi Koefisien Regresi Lokal GWR Antar Lokasi Aset:")
    print(f"{'ID ASET':<10} | {'KOORDINAT (LAT, LON)':<22} | {'BETA_1 (UMUR ASET)':<20} | {'BETA_2 (BEBAN)':<16} | {'PREDIKSI RUSAK'}")
    print("-" * 92)
    for idx in [0, 1, 2, N - 3, N - 2, N - 1]:
        a = assets[idx]
        coord = f"{a['lat']:.4f}, {a['lon']:.4f}"
        print(
            f"{a['asset_id']:<10} | {coord:<22} | {a['gwr_b1_age']:<20.4f} | {a['gwr_b2_load']:<16.4f} | {a['gwr_predicted']:.2f}"
        )
    print()


def spatial_kde_rf_tower_and_export(
    assets, filename="bengkulu_statistical_gis.geojson"
):
    print("=" * 85)
    print("TAHAP 3: SPATIAL KERNEL DENSITY ESTIMATION (KDE) & EKSPOR CLOUD GEOJSON")
    print("=" * 85)

    random.seed(99)
    blind_spots = [
        (random.gauss(-3.7980, 0.012), random.gauss(102.2620, 0.012))
        for _ in range(50)
    ] + [
        (random.gauss(-3.8220, 0.010), random.gauss(102.2900, 0.010))
        for _ in range(30)
    ]

    h_kde = 1.8  
    scored_candidates = []
    for pt in blind_spots:
        kde_sum = 0.0
        for other in blind_spots:
            d = haversine_km(pt[0], pt[1], other[0], other[1])
            if d <= h_kde:
                u = d / h_kde
                kde_sum += (15.0 / 16.0) * ((1.0 - u**2) ** 2)
        density_val = kde_sum / (len(blind_spots) * (h_kde**2))
        scored_candidates.append(
            {"lat": pt[0], "lon": pt[1], "kde_density": density_val}
        )

    scored_candidates.sort(key=lambda x: x["kde_density"], reverse=True)
    selected_towers = []
    for cand in scored_candidates:
        if all(
            haversine_km(cand["lat"], cand["lon"], t["lat"], t["lon"]) >= 2.0
            for t in selected_towers
        ):
            selected_towers.append(cand)
        if len(selected_towers) == 3:
            break

    print("Titik Optimal Pembangunan RF Repeater Tower Berdasarkan Puncak Kepadatan Spasial (KDE):")
    for idx, t in enumerate(selected_towers, 1):
        t["tower_id"] = f"RF-KDE-TOWER-0{idx}"
        print(
            f"- {t['tower_id']} -> Koordinat: ({t['lat']:.5f}, {t['lon']:.5f}) | Intensitas KDE: {t['kde_density']:.4f}"
        )

    features = []
    for a in assets:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [a["lon"], a["lat"]]},
            "properties": {
                "layer": "GWR_Public_Asset",
                "id": a["asset_id"],
                "damage_actual": round(a["damage_index"], 2),
                "damage_gwr_pred": round(a["gwr_predicted"], 2),
                "local_beta_age": round(a["gwr_b1_age"], 4),
            },
        })

    for t in selected_towers:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [t["lon"], t["lat"]]},
            "properties": {
                "layer": "Optimal_RF_Tower_KDE",
                "id": t["tower_id"],
                "kde_intensity": round(t["kde_density"], 4),
            },
        })

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {"type": "FeatureCollection", "features": features}, f, indent=4
        )

    print(f"\n[SUKSES] File spasial '{filename}' ({len(features)} fitur) berhasil dibuat untuk QGIS & GitHub!")


if __name__ == "__main__":
    spatial_assets = generate_spatial_assets_and_morans_i(n_assets=30)
    run_geographically_weighted_regression(spatial_assets, bandwidth_km=2.5)
    spatial_kde_rf_tower_and_export(spatial_assets)