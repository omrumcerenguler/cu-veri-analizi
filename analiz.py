import os
import sys
import time
import logging
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
import pandas as pd
import seaborn as sns
import numpy as np

from prophet import Prophet
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Çıktıların anında görünmesi için (buffering kapatma)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ---- STANDART PARAMETRELER ----
# "multi": her (HitId, Alan) çifti 1 sayılır; "single": yayın başı tek alan say
COUNT_STRATEGY = "multi"
YEAR_MIN = 2016
YEAR_MAX = 2024
TARGET_DEFAULT = 2026
# True yaparsan menü açılırken ekran temizlenir; False ise önceki çıktılar korunur
CLEAR_SCREEN = False

# Varsayılan tahmin modeli: "prophet" veya "lm"
MODEL_DEFAULT = "prophet"



# --- .env'den DB ayarlarını yükle ---
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
DB_ENCRYPT = os.getenv("DB_ENCRYPT", "Optional")  # Optional / Required / Strict
DB_TRUST = os.getenv("DB_TRUST_SERVER_CERT", "true").lower()  # true/false

# Driver adını ve kullanıcı/parolayı URL içinde güvenli kaçır
driver_q = quote_plus(DB_DRIVER)

connection_string = (
    f"mssql+pyodbc://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}@{DB_HOST}/{DB_NAME}"
    f"?driver={driver_q}&Encrypt={DB_ENCRYPT}&TrustServerCertificate={DB_TRUST}"
)

engine = create_engine(connection_string, pool_pre_ping=True, pool_recycle=1800, fast_executemany=True)

# --- Logging ve menü yardımcıları ---
# Prophet/cmdstanpy konsol loglarını azalt (menü çıktılarını bozmasın)

for _name in ("cmdstanpy", "prophet", "stanio"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

# --- Girdi yardımcıları: yanlış seçimlerde tekrar sor ---
def ask_choice(prompt, options, default=None):
    """
    options: liste, ör: ["1","2"] veya ["evet","hayır"]
    default: None değilse, kullanıcı boş ENTER yaparsa default döner
    """
    opts_lower = {str(o).lower(): o for o in options}
    while True:
        try:
            raw = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n⚠️ Girdi iptal edildi. Lütfen bir seçim yapın.")
            continue
        if raw == "" and default is not None:
            return default
        key = raw.lower()
        if key in opts_lower:
            return opts_lower[key]
        print(f"⚠️ Geçersiz seçim. Geçerli seçenekler: {', '.join(options)}")

def ask_yesno(prompt, default=None):
    """
    'e','h','evet','hayır','y','n' destekler. Dönen değer 'evet' veya 'hayır' olur.
    """
    mapping = {
        "e": "evet", "evet": "evet", "y": "evet", "yes": "evet",
        "h": "hayır", "hayır": "hayır", "n": "hayır", "no": "hayır"
    }
    while True:
        try:
            raw = input(prompt).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n⚠️ Girdi iptal edildi. Lütfen E/H yanıtı verin.")
            continue
        if raw == "" and default is not None:
            return default
        if raw in mapping:
            return mapping[raw]
        print("⚠️ Geçersiz seçim. Lütfen 'evet' veya 'hayır' (E/H) girin.")

def ask_int(prompt, default=None, min_val=None, max_val=None):
    while True:
        try:
            raw = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n⚠️ Girdi iptal edildi. Lütfen bir yıl girin.")
            continue
        if raw == "" and default is not None:
            val = int(default)
            return val
        try:
            val = int(raw)
        except ValueError:
            print("⚠️ Lütfen geçerli bir yıl girin (ör. 2026).")
            continue
        if min_val is not None and val < min_val:
            print(f"⚠️ En az {min_val} olabilir.")
            continue
        if max_val is not None and val > max_val:
            print(f"⚠️ En fazla {max_val} olabilir.")
            continue
        return val


def show_prediction_menu():
    """Menüyü temiz ekranda, kısa satırlarla ve güvenli biçimde gösterir."""
    if CLEAR_SCREEN:
        try:
            os.system("cls" if os.name == "nt" else "clear")
        except Exception:
            # Bazı ortamlarda clear çalışmayabilir; sorun değil.
            pass
    print("-" * 60, flush=True)
    print("Tahmin yapmak için aşağıdaki seçeneklerden birini seçiniz:", flush=True)
    print("1 — En çok yayına sahip ilk 10 alanın tahminini yap", flush=True)
    print("2 — Belirli bir alan ya da alanlar için tahmin yapmak istiyorum", flush=True)
    print("-" * 60, flush=True)


# -- 1. Sorgu: Yıllara göre toplam yayın sayısı
query = f"""
SELECT 
    wh.SourcePublishYear AS Yil,
    COUNT(*) AS YayinSayisi
FROM dbo.WOSHit wh
WHERE wh.SourcePublishYear BETWEEN {YEAR_MIN} AND {YEAR_MAX}
GROUP BY wh.SourcePublishYear
ORDER BY Yil;
"""
df = pd.read_sql(query, engine)
print("-" * 60)
print("Yıllara göre toplam yayın sayısı:")
print(df.head(10))

# Grafik
plt.figure(figsize=(10, 6))
plt.plot(df["Yil"], df["YayinSayisi"], marker='o')
plt.title("Yıllara Göre Toplam Yayın Sayısı")
plt.xlabel("Yıl")
plt.ylabel("Toplam Yayın")
plt.grid(True)
plt.savefig("yillara_gore_toplam_yayin_sayisi.png",
            dpi=300, bbox_inches='tight')
plt.close()

# --- 2. Sorgu: Birim bazlı yayın sayısı sorgusu ---
query_birim = f"""
SELECT 
    wh.SourcePublishYear AS Yil,
    yb.Ad AS Birim,
    COUNT(DISTINCT wh.HitId) AS YayinSayisi
FROM dbo.WOSHit wh
JOIN dbo.WosAuthor wa ON wh.HitId = wa.HitId
JOIN dbo.CuAuthorRID cr ON wa.ResearcherID = cr.ResearcherID
JOIN dbo.CuAuthor ca ON cr.CuAuthorID = ca.ID
JOIN dbo.YoksisBirim yb ON ca.YoksisId = yb.YoksisId
WHERE wh.SourcePublishYear BETWEEN {YEAR_MIN} AND {YEAR_MAX}
GROUP BY wh.SourcePublishYear, yb.Ad
ORDER BY wh.SourcePublishYear, YayinSayisi DESC;
"""
df_birim = pd.read_sql(query_birim, engine)
print("-" * 60)
print("Yıllara ve birimlere göre yayın sayısı:")
print(df_birim.head(10))

# Görsel: Yıllara ve birimlere göre yayın sayısı (en çok yayın yapan ilk 10 birim)
top_birimler = df_birim.groupby(
    "Birim")["YayinSayisi"].sum().nlargest(10).index
df_birim_filtered = df_birim[df_birim["Birim"].isin(top_birimler)]

plt.figure(figsize=(14, 8))
sns.barplot(
    data=df_birim_filtered,
    x="YayinSayisi",
    y="Birim",
    hue="Yil",
    palette="viridis"
)
plt.title("Yıllara Göre En Çok Yayın Yapan 10 Birim")
plt.xlabel("Yayın Sayısı")
plt.ylabel("Birim")
plt.legend(title="Yıl", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("yillara_gore_birim_yayin_sayisi.png", dpi=300)
plt.close()

# --- 3. Sorgu: En çok yayın yapan yazarlar ---
query_yazar = f"""
SELECT 
    ca.AdSoyad AS YazarAdSoyad,
    COUNT(DISTINCT wh.HitId) AS YayinSayisi
FROM dbo.WOSHit wh
JOIN dbo.WosAuthor wa ON wh.HitId = wa.HitId
JOIN dbo.CuAuthorRID cr ON wa.ResearcherID = cr.ResearcherID
JOIN dbo.CuAuthor ca ON cr.CuAuthorID = ca.ID
WHERE wh.SourcePublishYear BETWEEN {YEAR_MIN} AND {YEAR_MAX}
GROUP BY ca.AdSoyad
ORDER BY YayinSayisi DESC;
"""
df_yazar = pd.read_sql(query_yazar, engine)
print("-" * 60)
print("Yıllara ve birimlere göre yayın sayısı:")
print(df_yazar.head(10))

# Sadece en çok yayın yapan ilk 10 yazarı çiz
top_yazarlar = df_yazar.head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_yazarlar["YazarAdSoyad"],
         top_yazarlar["YayinSayisi"], color='skyblue')
plt.xlabel("Yayın Sayısı")
plt.title("En Çok Yayın Yapan 10 Yazar")
plt.gca().invert_yaxis()
plt.grid(axis='x')
plt.tight_layout()
plt.savefig("en_cok_yayin_yapan_yazarlar.png", dpi=300)
plt.close()

# --- 4. Sorgu: Alanlara göre yayın dağılımı ---
query_alan = f"""
SELECT 
    t.Value AS Alan,
    COUNT(*) AS YayinSayisi
FROM (
    SELECT DISTINCT wa.HitId, wa.Value
    FROM dbo.WosHitAttributes wa
    JOIN dbo.WOSHit wh ON wa.HitId = wh.HitId
    WHERE wa.Name = 'category_info.subject'
      AND wh.SourcePublishYear BETWEEN {YEAR_MIN} AND {YEAR_MAX}
) AS t
GROUP BY t.Value
ORDER BY YayinSayisi DESC;
"""
df_alan = pd.read_sql(query_alan, engine)
print("-" * 60)
print("Alanlara göre yayın sayısı:")
print(df_alan.head(10))

# Grafik
plt.figure(figsize=(10, 6))
plt.barh(df_alan["Alan"].head(10),
         df_alan["YayinSayisi"].head(10), color='lightgreen')
plt.xlabel("Yayın Sayısı")
plt.title("En Fazla Yayın Yapılan 10 Alan")
plt.gca().invert_yaxis()
plt.grid(axis='x')
plt.tight_layout()
plt.savefig("en_fazla_yayin_yapilan_alanlar.png", dpi=300)
plt.close()


# --- 5. Sorgu: Alanlara Göre Yıllık Yayın Sayıları (Trend Analizi için) ---
query_alan_yillik = f"""
SELECT 
    t.Yil,
    t.Alan,
    COUNT(*) AS YayinSayisi
FROM (
    SELECT DISTINCT 
        wh.SourcePublishYear AS Yil,
        wa.Value AS Alan,
        wa.HitId
    FROM dbo.WosHitAttributes wa
    JOIN dbo.WOSHit wh ON wa.HitId = wh.HitId
    WHERE wa.Name = 'category_info.subject'
      AND wh.SourcePublishYear BETWEEN {YEAR_MIN} AND {YEAR_MAX}
) AS t
GROUP BY t.Yil, t.Alan
ORDER BY t.Yil, YayinSayisi DESC;
"""


df_alan_yillik = pd.read_sql(query_alan_yillik, engine)

# Veri setindeki mevcut son yıl (tüm alanlar için)
DATA_LAST_YEAR = int(df_alan_yillik["Yil"].max())
ALLOWED_MIN_YEAR = DATA_LAST_YEAR + 1
ALLOWED_MAX_YEAR = DATA_LAST_YEAR + 100

# --- DEBUG: Chemistry yıllık seri ---
chem_py = (
    df_alan_yillik[df_alan_yillik["Alan"] == "Chemistry"]
    .groupby("Yil", as_index=False)["YayinSayisi"].sum()
    .sort_values("Yil")
)
print("DEBUG PY — Chemistry yıllık seri:")
print(chem_py.to_string(index=False))
chem_py.to_csv("debug_py_chemistry_yillik.csv", index=False)
print("-" * 60)

# Yeni sütun: Alan_Lower (küçük harfli alan adı)
df_alan_yillik["Alan_Lower"] = df_alan_yillik["Alan"].str.lower()

# Top 20 en çok yayına sahip alanı seç
top_alanlar = df_alan_yillik.groupby(
    "Alan")["YayinSayisi"].sum().nlargest(20).index
df_alan_yillik_filtered = df_alan_yillik[df_alan_yillik["Alan"].isin(
    top_alanlar)]

# Pivot tablo: satırlar yıllar, sütunlar alanlar, değerler yayın sayısı
alan_trend_df = df_alan_yillik_filtered.pivot(
    index="Yil", columns="Alan", values="YayinSayisi").fillna(0)

# İlk birkaç yılı yazdıralım
print("-" * 60)
print("Yıllara göre alan bazlı yayın sayısı:")
print(alan_trend_df.head(10))

# Görsel: Yıllara Göre En Çok Yayın Yapılan 20 Alan (Isı Haritası)
plt.figure(figsize=(14, 10))
sns.heatmap(alan_trend_df.T, cmap="YlGnBu", linewidths=.5,
            annot=True, fmt='g', cbar_kws={'label': 'Yayın Sayısı'})
plt.title("Yıllara Göre En Çok Yayın Yapılan 20 Alan (Isı Haritası)")
plt.xlabel("Yıl")
plt.ylabel("Alan")
plt.tight_layout()
plt.savefig("yillara_gore_en_cok_alan_yayin_sayisi_heatmap.png", dpi=300)
plt.close()

# -- 6. Sorgu: En çok atıf alan yazarları bul
query_atifli_yazarlar = """
SELECT 
    ca.AdSoyad AS Yazar,
    SUM(wh.CitiationCount) AS ToplamAtıf
FROM dbo.WosAuthor wa
JOIN dbo.WOSHit wh ON wa.HitId = wh.HitId
JOIN dbo.CuAuthorRID cr ON wa.ResearcherID = cr.ResearcherID
JOIN dbo.CuAuthor ca ON cr.CuAuthorID = ca.ID
WHERE wh.CitiationCount IS NOT NULL AND ca.AdSoyad IS NOT NULL
GROUP BY ca.AdSoyad
ORDER BY ToplamAtıf DESC;
"""

df_atifli_yazarlar = pd.read_sql(query_atifli_yazarlar, engine)
print("-" * 60)
print("Yıllara göre en çok atıf alan yazarlar:")
print(df_atifli_yazarlar.head(10))

# İlk 10 yazarı al
df_atifli_yazarlar_top10 = df_atifli_yazarlar.head(10)

# Görselleştir
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_atifli_yazarlar_top10,
    x="ToplamAtıf",
    y="Yazar",
    hue="Yazar",  # y eksenindeki değeri hue olarak da kullanıyoruz
    palette="magma",
    legend=False  # ekstra renk açıklaması çıkmaması için
)
plt.title("En Çok Atıf Alan 10 Yazar")
plt.xlabel("Toplam Atıf Sayısı")
plt.ylabel("Yazar")
plt.tight_layout()
plt.savefig("en_cok_atif_alan_yazarlar.png")
plt.close()

while True:
    show_prediction_menu()
    secim = ask_choice("Seçiminiz (1/2): ", ["1","2"])

    # Gerçek alan isimlerini küçük harfe çevirip eşleştirme için dict oluştur
    alan_mapping = {
        alan.lower(): alan for alan in df_alan_yillik["Alan"].unique()}

    if secim == "1":
        # Son 5 yıl verisini filtrele
        son_bes_yil = df_alan_yillik["Yil"].max() - 4
        df_son_5yil = df_alan_yillik[df_alan_yillik["Yil"] >= son_bes_yil]

        # Ortalama yayın sayılarına göre ilk 10 alan
        top_alanlar_tahmin = (
            df_son_5yil.groupby("Alan")["YayinSayisi"]
            .mean()
            .nlargest(10)
            .index
        )
        print("-" * 60)
        print("Son 5 yılın ortalama yayın sayısına göre ilk 10 alan:")
        for i, alan in enumerate(top_alanlar_tahmin, 1):
            print(f"{i}. {alan}")
        eslesen_alanlar = []
        for alan in top_alanlar_tahmin:
            alan_lower = alan.lower()
            for veri_alani in alan_mapping:
                if alan_lower == veri_alani or alan_lower in veri_alani:
                    eslesen_alanlar.append(alan_mapping[veri_alani])
                    break
    elif secim == "2":
        alan_goster = ask_yesno("Alanları bilmiyorsanız listeyi görmek ister misiniz? (E/H): ", default="hayır")
        if alan_goster == "evet":
            print("-" * 60)
            print("Tahmin yapılabilecek mevcut tüm alanlar aşağıda listelenmiştir:")
            print("-" * 60)
            print(df_alan["Alan"].tolist())
        while True:
            alan_input = input("Lütfen tahmin yapmak istediğiniz alan(lar)ı virgülle ayırarak giriniz:\n").strip()
            if alan_input:
                break
            print("⚠️ En az bir alan girmelisiniz.")
        top_alanlar_tahmin = [a.strip().lower()
                              for a in alan_input.split(",") if a.strip()]
    else:
        print("Geçersiz seçim. Lütfen 1 veya 2 giriniz.")
        print("-" * 60)
        time.sleep(0.8)  # kısa bekleme; doğrudan yeniden menüye dön
        continue

    # Kullanıcının girdikleriyle veri setindeki alanları eşleştir
    if secim != "1":
        eslesen_alanlar = []
        for kullanici_alani in top_alanlar_tahmin:
            for veri_alani in alan_mapping:
                if kullanici_alani == veri_alani:
                    eslesen_alanlar.append(alan_mapping[veri_alani])
                    break
                # Eğer kullanıcı Computer Science, Information Systems gibi virgüllü bir isme karşılık girdiyse
                elif kullanici_alani in veri_alani:
                    eslesen_alanlar.append(alan_mapping[veri_alani])
                    break

    if not eslesen_alanlar:
        print("⚠️ Hiçbir eşleşen alan bulunamadı. Lütfen tekrar deneyin.")
        print("-" * 60)
        time.sleep(0.6)
        continue  # menüye geri dön

    print("-" * 60)
    # Yıl, hemen burada doğrulansın (Prophet/Lm seçilmeden önce)
    _default_year = max(TARGET_DEFAULT, ALLOWED_MIN_YEAR)
    prompt_yil = f"Hangi yıl için tahmin yapılacak? ({ALLOWED_MIN_YEAR}–{ALLOWED_MAX_YEAR} | ENTER = {_default_year}): "
    hedef_yil = ask_int(prompt_yil, default=_default_year, min_val=ALLOWED_MIN_YEAR, max_val=ALLOWED_MAX_YEAR)

    # Model seçimi (prophet / lm)
    model_sec = ask_choice(f"Hangi model? (prophet/lm) [ENTER = {MODEL_DEFAULT}]: ", ["prophet","lm"], default=MODEL_DEFAULT)

    tahmin_sonuclari = []

    for alan in eslesen_alanlar:
        matching_alan = df_alan_yillik[df_alan_yillik["Alan"] == alan]
        if matching_alan.empty:
            print(f"⚠️ '{alan}' alanı bulunamadı, tahmin atlandı.")
            continue
        df_prophet = df_alan_yillik[df_alan_yillik["Alan"] == alan][[
            "Yil", "YayinSayisi"]]
        df_prophet["ds"] = pd.to_datetime(
            df_prophet["Yil"].astype(str), format="%Y")
        df_prophet = df_prophet.rename(columns={"YayinSayisi": "y"})
        if df_prophet["y"].count() < 2:
            print(
                f"⚠️ {alan} alanı için yeterli veri bulunamadığı için tahmin yapılamadı.")
            continue

        if model_sec == "lm":
            # Basit Doğrusal Regresyon (R'deki lm ile uyumlu)
            years = df_prophet["ds"].dt.year.values.astype(float)
            y_vals = df_prophet["y"].values.astype(float)

            if len(np.unique(years)) < 2:
                print(f"⚠️ {alan} alanı için yeterli farklı yıl yok, lm ile tahmin yapılamadı.")
                continue

            coef = np.polyfit(years, y_vals, 1)  # [slope, intercept]
            y_pred = np.polyval(coef, hedef_yil)
            y_pred = max(0.0, float(y_pred))

            # Kaydetme aşamasında grafik üretmek için gerekli verileri taşıyalım
            forecast = (years, y_vals, coef)
            model = "lm"
            tahmin = pd.DataFrame({"ds": [pd.to_datetime(f"{hedef_yil}")], "yhat": [y_pred]})

            tahmin_sonucu = round(y_pred)

        else:
            # Prophet
            model = Prophet(yearly_seasonality=False, daily_seasonality=False)
            model.fit(df_prophet)

            son_yil = df_prophet["ds"].dt.year.max()
            periods = hedef_yil - son_yil  # giriş aşamasında zaten aralık doğrulandı

            future = model.make_future_dataframe(periods=periods, freq='YS')
            forecast = model.predict(future)
            forecast["yhat"] = forecast["yhat"].clip(lower=0)

            tahmin = forecast[forecast["ds"].dt.year == hedef_yil]
            if tahmin.empty:
                print(f"⚠️ {alan} alanı için {hedef_yil} yılına ait geçerli tahmin üretilemedi.")
                continue
            tahmin_sonucu = round(tahmin["yhat"].values[0])

        print(
            f"Tahmine göre {alan} alanında {hedef_yil} yılında yaklaşık {tahmin_sonucu} yayın bekleniyor.")
        print("-" * 60)

        tahmin_sonuclari.append(
            (alan, hedef_yil, tahmin_sonucu, forecast, model, model_sec))

        print("-" * 60)

    if not tahmin_sonuclari:
        print("Tahmin yapılabilecek alan bulunamadı veya tahmin yapılamadı.")
        print("-" * 60)
    else:
        kaydet = ask_yesno("Tüm tahmin grafikleri kaydedilsin mi? (evet/hayır): ", default="hayır")
        if kaydet == "evet":
            for alan, yil, sonuc, forecast, model, model_sec in tahmin_sonuclari:
                plt.figure()
                if model_sec == "lm":
                    years, y_vals, coef = forecast
                    x_line = np.linspace(years.min()-0.5, max(years.max(), yil)+1, 100)
                    y_line = coef[0]*x_line + coef[1]
                    plt.scatter(years, y_vals)
                    plt.plot(x_line, y_line)
                    plt.scatter([yil], [sonuc], marker="x")
                    plt.title(f"{alan[:20]} - {yil} (LM) Tahmini")
                    plt.xlabel("Yıl")
                    plt.ylabel("Yayın Sayısı")
                else:
                    # Prophet
                    model.plot(forecast)
                    plt.title(f"{alan[:20]} - {yil} (Prophet) Tahmini")
                    plt.xlabel("Yıl")
                    plt.ylabel("Yayın Sayısı")
                plt.savefig(f"{alan[:20]}_{yil}_tahmin.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"{alan[:20]}_{yil}_tahmin.png dosyası kaydedildi.")
        else:
            print("Grafikler kaydedilmedi.")
            print("-" * 60)

    print("Tahmin işlemi tamamlandı.")
    print("-" * 60)

    # Yeniden başlatma veya çıkış seçeneği
    tekrar = ask_yesno("Başka bir alan için tahmin yapmak ister misiniz? (evet/hayır): ", default="hayır")
    if tekrar == "evet":
        print("Program yeniden başlatılıyor...", flush=True)
        time.sleep(0.6)  # kısa bekleme; ekran temizlenecek
        continue  # Döngü başına dön; sadece menü yeniden gösterilecek
    else:
        print("Program sonlandırıldı.")
        print("-" * 60)
        exit()
