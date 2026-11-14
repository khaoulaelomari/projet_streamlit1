# -*- coding: utf-8 -*-
"""
Prédiction des attaques classiques en cyber — application Streamlit
Auteure : EL OMARI Khaoula

Reprend la logique du fichier projet_10_attaques_ml.py :
  doublons -> NaN (fillna(0)) -> X / y (sans identifiants ni attack_type)
  -> get_dummies -> split 70/15/15 stratifié -> StandardScaler (fit sur train)
  -> 5 modèles comparés sur la validation (F1) -> évaluation finale sur le test.
La différence : un seul modèle multi-classes (Normal + 10 attaques) au lieu de
10 modèles binaires (c'est le « défi bonus n°4 » du notebook).

Lancer :  streamlit run app.py
"""
import os
import time
import hmac
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
import sklearn
import xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ───────────────────────────── Identité & accès ─────────────────────────────
NOM_COMPLET = "EL OMARI Khaoula"
UTILISATEUR = "Khaoula"
MOT_DE_PASSE = "Ke.2005@"

DOSSIER = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(DOSSIER, "dataset.csv")
CACHE = os.path.join(DOSSIER, ".cache_modeles.joblib")

# ───────────────────────────── Constantes du projet ─────────────────────────
ATTAQUES = ["DDoS", "Phishing", "SQL_Injection", "Brute_Force", "XSS", "Malware",
            "Ransomware", "Port_Scanning", "Man_in_the_Middle", "Botnet"]
CLASSES = ["Normal"] + ATTAQUES
AFF = {"SQL_Injection": "SQL Injection", "Brute_Force": "Brute Force",
       "Port_Scanning": "Port Scanning", "Man_in_the_Middle": "Man-in-the-Middle"}
def aff(c): return AFF.get(c, c)

COULEURS = {"Normal": "#22C55E", "DDoS": "#7C3AED", "Phishing": "#EC4899", "SQL_Injection": "#3B82F6",
            "Brute_Force": "#F59E0B", "XSS": "#10B981", "Malware": "#EF4444", "Ransomware": "#14B8A6",
            "Port_Scanning": "#6366F1", "Man_in_the_Middle": "#F97316", "Botnet": "#A855F7"}

COLS_IDENTIFIANTS = ["sample_id", "timestamp_utc", "src_ip", "dst_ip", "src_port", "attack_type"]
CATEGORIES = ["protocol", "service", "dst_port"]
ORDRE_MODELES = ["Régression linéaire", "Régression logistique", "SVM", "Random Forest", "XGBoost"]
MODELES_PROBA = ["Régression logistique", "Random Forest", "XGBoost"]
N_SVM = 10000

# variables numériques : (libellé, groupe, unité)
VARS = {
    "avg_packet_size_bytes": ("Taille du paquet (Packet Size)", "main", "octets"),
    "packet_rate_pps": ("Taux de paquets (Packet Rate)", "main", "pkt/s"),
    "flow_duration_ms": ("Durée (Duration)", "main", "ms"),
    "failed_logins_5m": ("Tentatives échouées (Failed Attempts)", "main", ""),
    "payload_entropy": ("Entropie de la charge utile (Payload Entropy)", "main", ""),
    "packet_count": ("Nombre de paquets", "Volumétrie & TCP", ""),
    "byte_count": ("Nombre d’octets", "Volumétrie & TCP", "octets"),
    "byte_rate_Bps": ("Débit (octets/s)", "Volumétrie & TCP", "o/s"),
    "syn_count": ("Paquets SYN", "Volumétrie & TCP", ""),
    "ack_count": ("Paquets ACK", "Volumétrie & TCP", ""),
    "rst_count": ("Paquets RST", "Volumétrie & TCP", ""),
    "fin_count": ("Paquets FIN", "Volumétrie & TCP", ""),
    "unique_dst_ports_5m": ("Ports distincts (5 min)", "Balayage & portée", ""),
    "unique_dst_ips_5m": ("IP destination distinctes (5 min)", "Balayage & portée", ""),
    "http_requests_5m": ("Requêtes HTTP (5 min)", "Web", ""),
    "http_4xx_5m": ("Erreurs HTTP 4xx", "Web", ""),
    "http_5xx_5m": ("Erreurs HTTP 5xx", "Web", ""),
    "sql_keyword_hits_5m": ("Mots-clés SQL détectés", "Web", ""),
    "xss_pattern_hits_5m": ("Motifs XSS détectés", "Web", ""),
    "url_length": ("Longueur de l’URL", "Web", ""),
    "dns_queries_5m": ("Requêtes DNS (5 min)", "DNS & C2", ""),
    "dns_nxdomain_5m": ("Réponses DNS NXDOMAIN", "DNS & C2", ""),
    "outbound_connections_5m": ("Connexions sortantes (5 min)", "DNS & C2", ""),
    "beacon_interval_cv": ("Régularité des balises (CV)", "DNS & C2", ""),
    "arp_spoof_events_5m": ("Événements ARP spoofing", "Interception (MitM)", ""),
    "tls_cert_errors_5m": ("Erreurs de certificat TLS", "Interception (MitM)", ""),
    "email_links_count": ("Liens dans l’e-mail", "E-mail (phishing)", ""),
    "sender_domain_age_days": ("Âge du domaine expéditeur (jours)", "E-mail (phishing)", "j"),
    "spf_fail": ("Échec SPF (0 = non, 1 = oui)", "E-mail (phishing)", ""),
    "process_spawns_5m": ("Processus lancés (5 min)", "Hôte (malware / ransomware)", ""),
    "suspicious_executables_5m": ("Exécutables suspects", "Hôte (malware / ransomware)", ""),
    "file_modifications_5m": ("Modifications de fichiers", "Hôte (malware / ransomware)", ""),
    "file_entropy": ("Entropie des fichiers", "Hôte (malware / ransomware)", ""),
    "outbound_byte_ratio": ("Ratio d’octets sortants", "Hôte (malware / ransomware)", ""),
}
NUM = list(VARS)
PROTOCOLES = ["TCP", "UDP"]
SERVICES = ["HTTP", "HTTPS", "DNS", "SMTP", "NTP", "SSH"]

PAGES = [("Tableau de bord", "space_dashboard"), ("Prédiction / Test", "science"),
         ("Analyses", "monitoring"), ("Incidents", "warning"),
         ("Données & Modèle", "database"), ("Évaluation", "verified"), ("Paramètres", "settings")]

# ───────────────────────────── Utilitaires ──────────────────────────────────
def H(s):
    return "".join(l.strip() for l in s.splitlines())

def nf(x, d=0):
    return f"{x:,.{d}f}".replace(",", "\u00a0").replace(".", ",")

def pc(x, d=2):
    return nf(100 * x, d) + "\u00a0%"

LOGO_SVG = """<svg width="{s}" height="{s}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="lg{u}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#A78BFA"/><stop offset="1" stop-color="#5B21B6"/></linearGradient></defs>
<path d="M32 3 55 11v19c0 15-10 26-23 31C19 56 9 45 9 30V11z" fill="url(#lg{u})"/>
<g stroke="#fff" stroke-width="2.2" stroke-linecap="round" fill="none" opacity=".95">
<path d="M32 20v8M32 28l-9 6M32 28l9 6M23 34v8M41 34v8M23 42l9 5 9-5M32 47v-4"/></g>
<g fill="#fff"><circle cx="32" cy="19" r="3.4"/><circle cx="23" cy="34" r="3"/><circle cx="41" cy="34" r="3"/>
<circle cx="23" cy="43" r="3"/><circle cx="41" cy="43" r="3"/><circle cx="32" cy="29" r="2.4"/></g></svg>"""

def logo(s=44, u="a"):
    return LOGO_SVG.format(s=s, u=u)

# ───────────────────────────── Données & entraînement ───────────────────────
def prefixe_dummies(X):
    return pd.get_dummies(X, columns=CATEGORIES, dtype=int)

def entrainer(df_brut):
    t0 = time.time()
    df = df_brut.copy()
    colonnes = [c for c in df.columns if c != "sample_id"]
    n_doublons = int(df.duplicated(subset=colonnes).sum())
    df = df.drop_duplicates(subset=colonnes).reset_index(drop=True)
    n_nan = int(df.isna().sum().sum())
    nan_cols = df.isna().sum()
    nan_cols = nan_cols[nan_cols > 0]

    # profils par classe (avant fillna) : médiane / mode, + plages des curseurs
    profils, modes = {}, {}
    for c in CLASSES:
        sub = df[df["attack_type"] == c]
        profils[c] = sub[NUM].median()
        modes[c] = {k: sub[k].mode().iloc[0] for k in CATEGORIES}
    plages = {}
    for f in NUM:
        s = df[f].dropna()
        est_int = pd.api.types.is_integer_dtype(df[f])
        mn, mx = float(s.min()), float(s.quantile(0.999)) if s.max() > 3 * s.quantile(0.999) else float(s.max())
        plages[f] = (mn, mx, bool(est_int))

    # signatures (étape 05 du notebook) : écart moyen attaque/normal en σ
    dfn = df.fillna(0)
    sig = {}
    colsig = [c for c in dfn.columns if c in NUM]
    sd = dfn[colsig].std().replace(0, 1)
    mu_norm = dfn.loc[dfn.attack_type == "Normal", colsig].mean()
    for a in ATTAQUES:
        ec = ((dfn.loc[dfn.attack_type == a, colsig].mean() - mu_norm) / sd).abs().sort_values(ascending=False).head(4)
        sig[a] = ", ".join(f"{k} ({v:.1f}σ)" for k, v in ec.items())

    # X / y
    y = dfn["attack_type"].map({c: i for i, c in enumerate(CLASSES)}).values
    X = dfn.drop(columns=["is_attack"] + COLS_IDENTIFIANTS)
    X = prefixe_dummies(X).astype("float32")
    cols = list(X.columns)

    idx = np.arange(len(X))
    i_tr, i_tmp = train_test_split(idx, test_size=0.30, random_state=42, stratify=y)
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, random_state=42, stratify=y[i_tmp])
    Xtr, Xva, Xte = X.iloc[i_tr], X.iloc[i_va], X.iloc[i_te]
    ytr, yva, yte = y[i_tr], y[i_va], y[i_te]

    scaler = StandardScaler().fit(Xtr)
    Str, Sva, Ste = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    modeles, pred_val, duree = {}, {}, {}
    def chrono(nom, f):
        t = time.time(); f(); duree[nom] = time.time() - t

    def f_lin():
        m = LinearRegression().fit(Str, np.eye(len(CLASSES))[ytr])
        modeles["Régression linéaire"] = m
        pred_val["Régression linéaire"] = m.predict(Sva).argmax(1)
    def f_log():
        m = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced").fit(Str, ytr)
        modeles["Régression logistique"] = m
        pred_val["Régression logistique"] = m.predict(Sva)
    def f_svm():
        Xs, _, ys, _ = train_test_split(Str, ytr, train_size=min(N_SVM, len(Str)), random_state=42, stratify=ytr)
        m = SVC(kernel="rbf", C=1.0, class_weight="balanced").fit(Xs, ys)
        modeles["SVM"] = m
        pred_val["SVM"] = m.predict(Sva)
    def f_rf():
        m = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                                   class_weight="balanced_subsample").fit(Xtr, ytr)
        modeles["Random Forest"] = m
        pred_val["Random Forest"] = m.predict(Xva)
    def f_xgb():
        m = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric="mlogloss",
                          random_state=42, n_jobs=-1, tree_method="hist")
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        modeles["XGBoost"] = m
        pred_val["XGBoost"] = m.predict(Xva)
    for nom, f in [("Régression linéaire", f_lin), ("Régression logistique", f_log), ("SVM", f_svm),
                   ("Random Forest", f_rf), ("XGBoost", f_xgb)]:
        chrono(nom, f)

    lignes = []
    for nom in ORDRE_MODELES:
        p = pred_val[nom]
        lignes.append({"Modèle": nom, "Accuracy": accuracy_score(yva, p),
                       "Précision": precision_score(yva, p, average="macro", zero_division=0),
                       "Recall": recall_score(yva, p, average="macro", zero_division=0),
                       "F1-score": f1_score(yva, p, average="macro", zero_division=0),
                       "Temps (s)": duree[nom]})
    tab_val = pd.DataFrame(lignes).set_index("Modèle")
    meilleur_global = tab_val["F1-score"].idxmax()
    deploye = tab_val.loc[MODELES_PROBA, "F1-score"].idxmax()   # modèle qui fournit des probabilités

    def proba(nom, Xr, Xs):
        m = modeles[nom]
        return m.predict_proba(Xs if nom == "Régression logistique" else Xr)
    P_te = proba(deploye, Xte, Ste)
    pr = P_te.argmax(1)
    cm = confusion_matrix(yte, pr, labels=range(len(CLASSES)))
    vrai_b, pred_b = yte > 0, pr > 0
    tn = int(((~vrai_b) & (~pred_b)).sum()); fp = int(((~vrai_b) & pred_b).sum())
    test = {
        "Accuracy": accuracy_score(yte, pr),
        "Précision": precision_score(yte, pr, average="macro", zero_division=0),
        "Recall": recall_score(yte, pr, average="macro", zero_division=0),
        "F1-score": f1_score(yte, pr, average="macro", zero_division=0),
        "Spécificité": tn / max(tn + fp, 1),
        "AUC": roc_auc_score(yte, P_te, multi_class="ovr", average="macro"),
        "Détection (attaque vs normal) — Précision": precision_score(vrai_b, pred_b, zero_division=0),
        "Détection (attaque vs normal) — Recall": recall_score(vrai_b, pred_b, zero_division=0),
        "Détection (attaque vs normal) — F1": f1_score(vrai_b, pred_b, zero_division=0),
    }
    rapport = pd.DataFrame(classification_report(yte, pr, target_names=CLASSES, output_dict=True,
                                                 zero_division=0)).T.iloc[:len(CLASSES)]
    rapport.index = [aff(c) for c in rapport.index]

    meta = dfn.iloc[i_te][["timestamp_utc", "src_ip", "dst_ip", "protocol", "service", "dst_port",
                           "packet_count"]].copy()
    meta["vrai"] = [CLASSES[i] for i in yte]
    meta["pred"] = [CLASSES[i] for i in pr]
    meta["conf"] = P_te.max(1)
    meta["heure"] = pd.to_datetime(meta["timestamp_utc"]).dt.hour
    imp = pd.Series(modeles["Random Forest"].feature_importances_, index=cols).sort_values(ascending=False)

    return dict(
        modeles=modeles, scaler=scaler, cols=cols, deploye=deploye, meilleur_global=meilleur_global,
        tab_val=tab_val, test=test, rapport=rapport, cm=cm, meta=meta, P_te=P_te,
        profils=profils, modes=modes, plages=plages, sig=sig, imp=imp,
        info=dict(n_brut=len(df_brut), n_doublons=n_doublons, n_nan=n_nan, nan_cols=nan_cols.to_dict(),
                  n_final=len(df), n_train=len(i_tr), n_val=len(i_va), n_test=len(i_te),
                  n_features=len(cols), duree=time.time() - t0),
        versions=(sklearn.__version__, xgboost.__version__, os.path.getsize(CSV)),
    )

@st.cache_data(show_spinner=False)
def charger_csv():
    return pd.read_csv(CSV)

@st.cache_resource(show_spinner=False)
def obtenir_modeles():
    df = charger_csv()
    sig = (sklearn.__version__, xgboost.__version__, os.path.getsize(CSV))
    if os.path.exists(CACHE):
        try:
            R = joblib.load(CACHE)
            if R.get("versions") == sig:
                return R
        except Exception:
            pass
    R = entrainer(df)
    try:
        joblib.dump(R, CACHE)
    except Exception:
        pass
    return R

def predire_proba(R, X):
    """X : DataFrame déjà aligné sur R['cols']."""
    nom = R["deploye"]
    m = R["modeles"][nom]
    if nom == "Régression logistique":
        return m.predict_proba(R["scaler"].transform(X))
    return m.predict_proba(X)

def aligner(R, d):
    d = prefixe_dummies(d)
    return d.reindex(columns=R["cols"], fill_value=0).astype("float32")

# ───────────────────────────── Style ────────────────────────────────────────
def injecter_css():
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea { font-family: 'Plus Jakarta Sans', 'Segoe UI', system-ui, sans-serif !important; }
.stApp { background: linear-gradient(180deg, #F3EFFF 0%, #FAF9FF 45%, #F6F3FF 100%); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px; }
h1, h2, h3, h4 { color: #1E1B4B; letter-spacing: -0.01em; }

[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #E9E3FF; min-width: 250px !important; }
[data-testid="stSidebar"] > div:first-child { padding-top: .6rem; }
.brand { display:flex; gap:.6rem; align-items:center; padding: .2rem .4rem 1rem; border-bottom: 1px solid #F0EBFF; margin-bottom: 1rem; }
.brand .t1 { font-weight: 800; font-size: 1.12rem; color:#4C1D95; line-height:1.1; }
.brand .t2 { font-size: .62rem; color:#7C3AED; letter-spacing:.04em; margin-top:2px; }
[data-testid="stSidebar"] div[role="radiogroup"] { gap: .25rem; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"], [data-testid="stSidebar"] div[role="radiogroup"] > label { padding: .62rem .85rem; border-radius: 12px; cursor: pointer; width: 100%; transition: background .15s; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child, [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"] p, [data-testid="stSidebar"] div[role="radiogroup"] > label p { font-weight: 600; font-size: .93rem; color: #4B4579; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover { background: #F5F0FF; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"], [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) { background: linear-gradient(135deg,#7C3AED,#5B21B6); box-shadow: 0 6px 16px rgba(91,33,182,.28); }
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p, [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p { color: #fff; }

[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border: 1px solid #ECE6FF !important; border-radius: 18px !important; box-shadow: 0 6px 22px rgba(109,40,217,.07); }
.ct { font-weight: 700; color:#1E1B4B; font-size: .98rem; display:flex; align-items:center; gap:.5rem; margin-bottom:.1rem; }
.ct .dot { width: 22px; height: 22px; border-radius: 7px; background:#EDE5FF; color:#6D28D9; display:inline-flex; align-items:center; justify-content:center; font-size: .8rem; }
.hd { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom: 1rem; flex-wrap: wrap; }
.hd .l { display:flex; align-items:center; gap:.9rem; }
.hd h1 { margin:0; font-size: 1.65rem; font-weight: 800; color:#2E1065; }
.hd .sub { color:#6B6494; font-size:.82rem; margin-top:.15rem; }
.hd .r { display:flex; align-items:center; gap:.7rem; }
.ic { width:34px; height:34px; border-radius:50%; background:#fff; border:1px solid #E9E3FF; display:flex; align-items:center; justify-content:center; color:#6D28D9; font-size:.95rem; }
.user { display:flex; align-items:center; gap:.55rem; background:#fff; border:1px solid #E9E3FF; padding:.3rem .8rem .3rem .35rem; border-radius: 999px; font-size:.82rem; font-weight:600; color:#2E1065; }
.user .av { width:28px; height:28px; border-radius:50%; background: linear-gradient(135deg,#8B5CF6,#5B21B6); color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700; }

.kpi { background:#fff; border:1px solid #ECE6FF; border-radius:18px; padding:.95rem 1.05rem; display:flex; gap:.85rem; align-items:center; box-shadow: 0 6px 22px rgba(109,40,217,.07); height: 100%; }
.kpi .ico { width:46px; height:46px; border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:1.3rem; flex-shrink:0; }
.kpi .lb { font-size:.74rem; color:#6B6494; font-weight:600; }
.kpi .v { font-size:1.55rem; font-weight:800; color:#1E1B4B; line-height:1.15; }
.kpi .d { font-size:.7rem; color:#8B83B5; }

.leg { display:flex; flex-direction:column; gap:.55rem; justify-content:center; height:100%; padding-top:.5rem; }
.leg .row { display:flex; align-items:center; justify-content:space-between; font-size:.8rem; color:#4B4579; gap:.6rem; }
.leg .row b { color:#1E1B4B; }
.leg .dt { width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:.45rem; }

table.inc { width:100%; border-collapse: collapse; font-size:.8rem; }
table.inc th { text-align:left; color:#8B83B5; font-weight:600; font-size:.72rem; padding:.35rem .5rem; border-bottom:1px solid #F0EBFF; }
table.inc td { padding:.5rem .5rem; border-bottom:1px solid #F7F4FF; color:#2E2A55; }
.badge { padding:.12rem .6rem; border-radius:6px; font-size:.72rem; font-weight:700; }
.b-att { background:#FDE8E8; color:#C81E1E; } .b-ok { background:#DCFCE7; color:#15803D; }

.res { border:1.5px solid; border-radius:16px; padding:.9rem 1rem; display:flex; align-items:center; gap:1rem; margin-top:.6rem; }
.res.att { border-color:#FCA5A5; background:#FFF5F5; } .res.ok { border-color:#86EFAC; background:#F3FFF7; }
.res .big { font-size:1.9rem; font-weight:800; line-height:1; }
.res.att .big { color:#DC2626; } .res.ok .big { color:#16A34A; }
.res .cf { margin-left:auto; text-align:right; min-width: 120px; }
.res .cf .n { font-size:1.35rem; font-weight:800; }
.bar { height:7px; border-radius:99px; background:#EEE; overflow:hidden; margin-top:.3rem; }
.bar > div { height:100%; border-radius:99px; }
.chip { display:inline-block; padding:.2rem .7rem; border-radius:999px; font-weight:700; font-size:.8rem; color:#fff; }
.small { font-size:.78rem; color:#6B6494; }

div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primaryFormSubmit"], div.stFormSubmitButton > button { background: linear-gradient(135deg,#7C3AED,#5B21B6); border: none; color:#fff; font-weight:700; border-radius: 12px; padding:.55rem 1rem; box-shadow:0 8px 18px rgba(91,33,182,.28); }
div.stButton > button[kind="primary"]:hover { filter: brightness(1.08); color:#fff; }
div.stButton > button[kind="secondary"] { border-radius: 12px; border:1px solid #DDD0FF; color:#5B21B6; font-weight:600; }
[data-testid="stSlider"] label p, [data-testid="stSelectbox"] label p, [data-testid="stNumberInput"] label p { font-size:.76rem !important; font-weight:600; color:#4B4579; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { color:#6D28D9; font-weight:700; }
div[role="slider"] { background-color:#7C3AED !important; box-shadow: 0 0 0 4px rgba(124,58,237,.18) !important; }
[data-testid="stExpander"] { border:1px solid #ECE6FF; border-radius:14px; background:#FBFAFF; }
.login-card { text-align:center; padding: .5rem 0 .2rem; }
.login-card h2 { margin:.6rem 0 .1rem; color:#2E1065; font-weight:800; }
</style>""", unsafe_allow_html=True)

# ───────────────────────────── Composants visuels ───────────────────────────
def entete(titre, sous):
    st.markdown(H(f"""<div class="hd"><div class="l">
      <div style="width:54px;height:54px;border-radius:16px;background:#EDE5FF;display:flex;align-items:center;justify-content:center">{logo(34, 'h')}</div>
      <div><h1>{titre}</h1><div class="sub">{sous}</div></div></div>
      <div class="r"><div class="ic">🔔</div><div class="ic">?</div>
      <div class="user"><div class="av">{NOM_COMPLET.split()[-1][0]}</div>{NOM_COMPLET} ⌄</div></div></div>"""),
                unsafe_allow_html=True)

def titre_carte(icone, texte):
    st.markdown(f'<div class="ct"><span class="dot">{icone}</span>{texte}</div>', unsafe_allow_html=True)

def kpi(icone, couleur_fond, couleur, label, valeur, detail):
    return H(f"""<div class="kpi"><div class="ico" style="background:{couleur_fond};color:{couleur}">{icone}</div>
      <div><div class="lb">{label}</div><div class="v">{valeur}</div><div class="d">{detail}</div></div></div>""")

def style_fig(fig, h=240, legend=False):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      font=dict(family="Plus Jakarta Sans, Segoe UI, sans-serif", size=11, color="#4B4579"))
    return fig

def table_incidents(df):
    lignes = ""
    for i, (_, r) in enumerate(df.iterrows(), 1):
        att = r["pred"] != "Normal"
        lignes += (f"<tr><td>{i}</td><td>{r['timestamp_utc']}</td><td>{r['src_ip']}</td><td>{r['dst_ip']}</td>"
                   f"<td><b style='color:{COULEURS[r['pred']]}'>{aff(r['pred'])}</b></td><td>{pc(r['conf'])}</td>"
                   f"<td><span class='badge {'b-att' if att else 'b-ok'}'>{'Attaque' if att else 'Normal'}</span></td></tr>")
    return H("<table class='inc'><tr><th>#</th><th>Horodatage</th><th>Adresse source</th><th>Adresse destination</th>"
             "<th>Classe prédite</th><th>Probabilité</th><th>Résultat</th></tr>") + lignes + "</table>"

# ───────────────────────────── Panneau de test ──────────────────────────────
def _sync(nom, extra=None):
    st.session_state.P[nom] = st.session_state[f"w_{nom}"]
    if extra:
        extra()

def champ(kind, nom, label, extra=None, **kw):
    k = f"w_{nom}"
    if k not in st.session_state:
        st.session_state[k] = st.session_state.P[nom]
    return getattr(st, kind)(label, key=k, on_change=_sync, args=(nom, extra), **kw)

def valeur_scenario(R, sc, niveau):
    base, cible = R["profils"]["Normal"], R["profils"][sc]
    out = {}
    for f in NUM:
        mn, mx, est_int = R["plages"][f]
        v = float(np.clip(base[f] + niveau * (cible[f] - base[f]), mn, mx))
        out[f] = int(round(v)) if est_int else (round(v) * 1.0 if (mx - mn) > 100 else round(v, 2))
    m = R["modes"][sc if niveau >= 0.5 else "Normal"]
    out.update(proto=m["protocol"], service=m["service"], port=int(m["dst_port"]))
    return out

def appliquer_scenario():
    R = obtenir_modeles()
    P = st.session_state.P
    P["scenario"], P["suspicion"] = st.session_state["w_scenario"], st.session_state["w_suspicion"]
    for k, v in valeur_scenario(R, P["scenario"], P["suspicion"]).items():
        P[k] = v
        st.session_state[f"w_{k}"] = v

def init_etat(R):
    if "P" not in st.session_state:
        st.session_state.P = {"scenario": "DDoS", "suspicion": 0.78, "sim": True, "log": True}
        st.session_state.P.update(valeur_scenario(R, "DDoS", 0.78))
        st.session_state.journal = []
        st.session_state.dernier = None

def slider_var(R, f, extra=None):
    mn, mx, est_int = R["plages"][f]
    lib, _, unite = VARS[f]
    lib = lib + (f" — {unite}" if unite and f in ("avg_packet_size_bytes", "packet_rate_pps", "flow_duration_ms") else "")
    if est_int:
        mn, mx = int(mn), int(np.ceil(mx))
        step = max(1, (mx - mn) // 400)
        champ("slider", f, lib, min_value=mn, max_value=mx, step=step)
    else:
        if mx - mn > 100:
            mn, mx = float(int(mn)), float(int(np.ceil(mx)))
            champ("slider", f, lib, min_value=mn, max_value=mx, step=float(max(1, int((mx - mn) // 400))), format="%.0f")
        else:
            mn, mx = float(round(mn, 2)), float(round(mx + 0.005, 2))
            champ("slider", f, lib, min_value=mn, max_value=mx, step=0.01, format="%.2f")

def panneau_test(R, large=False):
    """Affiche les paramètres modifiables, le bouton Prédire et le résultat."""
    ordre_att = [c for c in CLASSES]
    c1, c2 = st.columns(2)
    with c1:
        champ("selectbox", "scenario", "Scénario type (charge un profil)", extra=appliquer_scenario,
              options=ordre_att, format_func=aff)
    with c2:
        champ("slider", "suspicion", "Niveau de suspicion", extra=appliquer_scenario,
              min_value=0.0, max_value=1.0, step=0.01,
              help="Intensité du scénario chargé : 0 = trafic normal, 1 = profil complet de l’attaque choisie.")
    c1, c2 = st.columns(2)
    with c1:
        champ("selectbox", "proto", "Protocole", options=PROTOCOLES)
    with c2:
        champ("number_input", "port", "Port", min_value=1, max_value=65535, step=1)
    c1, c2 = st.columns(2)
    with c1: slider_var(R, "avg_packet_size_bytes")
    with c2: slider_var(R, "packet_rate_pps")
    c1, c2 = st.columns(2)
    with c1: slider_var(R, "flow_duration_ms")
    with c2: slider_var(R, "failed_logins_5m")
    c1, c2 = st.columns(2)
    with c1: champ("selectbox", "service", "Service de destination (Destination Service)", options=SERVICES)
    with c2: slider_var(R, "payload_entropy")
    c1, c2 = st.columns(2)
    with c1: champ("toggle", "sim", "Mode simulation", help="Ajoute une petite variabilité aléatoire réaliste (≈ 3 %) aux paramètres.")
    with c2: champ("toggle", "log", "Journaliser le test")

    groupes = {}
    for f, (_, g, _) in VARS.items():
        if g != "main":
            groupes.setdefault(g, []).append(f)
    with st.expander(f"Paramètres avancés ({sum(len(v) for v in groupes.values())} variables)", expanded=large):
        for g, fs in groupes.items():
            st.markdown(f"**{g}**")
            cols = st.columns(3 if large else 2)
            for i, f in enumerate(fs):
                with cols[i % len(cols)]:
                    slider_var(R, f)

    if st.button("▷  Prédire", type="primary", width="stretch", key="btn_predire" + ("_L" if large else "")):
        lancer_prediction(R)
    afficher_resultat(R)

def lancer_prediction(R):
    P = st.session_state.P
    rng = np.random.default_rng()
    ligne = {}
    for f in NUM:
        v = float(P[f])
        if P["sim"]:
            v *= 1 + rng.normal(0, 0.03)
        mn, mx, est_int = R["plages"][f]
        v = max(v, mn)
        ligne[f] = round(v) if est_int else v
    ligne.update(protocol=P["proto"], service=P["service"], dst_port=int(P["port"]))
    X = aligner(R, pd.DataFrame([ligne]))
    pr = predire_proba(R, X)[0]
    ordre = np.argsort(pr)[::-1]
    res = dict(classe=CLASSES[ordre[0]], conf=float(pr[ordre[0]]),
               top3=[(CLASSES[i], float(pr[i])) for i in ordre[:3]], heure=time.strftime("%Y-%m-%d %H:%M:%S"))
    st.session_state.dernier = res
    if P["log"]:
        st.session_state.journal.insert(0, {
            "Horodatage": res["heure"], "Protocole": P["proto"], "Port": int(P["port"]), "Service": P["service"],
            "Taille paquet": P["avg_packet_size_bytes"], "Taux paquets": P["packet_rate_pps"],
            "Durée (ms)": P["flow_duration_ms"], "Échecs connexion": P["failed_logins_5m"],
            "Entropie": P["payload_entropy"], "Suspicion": P["suspicion"],
            "Classe prédite": aff(res["classe"]), "Confiance": round(res["conf"], 4),
            "Résultat": "Normal" if res["classe"] == "Normal" else "Attaque"})

def afficher_resultat(R):
    r = st.session_state.dernier
    if r is None:
        st.markdown('<div class="small" style="margin-top:.6rem">Réglez les paramètres puis cliquez sur « Prédire ».</div>',
                    unsafe_allow_html=True)
        return
    att = r["classe"] != "Normal"
    coul = "#DC2626" if att else "#16A34A"
    top = " • ".join(f"{aff(c)} ({pc(p)})" for c, p in r["top3"])
    st.markdown(H(f"""<div class="res {'att' if att else 'ok'}">
      <div style="font-size:2rem">{'🛡️' if att else '✅'}</div>
      <div><div class="small">Résultat de la prédiction</div><div class="big">{'Attaque' if att else 'Normal'}</div></div>
      <div class="cf"><div class="small">Confiance</div><div class="n" style="color:{coul}">{pc(r['conf'])}</div>
      <div class="bar"><div style="width:{r['conf']*100:.1f}%;background:{coul}"></div></div></div></div>
      <div style="display:flex;gap:1.2rem;align-items:center;margin-top:.7rem;flex-wrap:wrap">
      <div><span class="small">Classe prédite</span><br><span class="chip" style="background:{COULEURS[r['classe']]}">{aff(r['classe'])}</span></div>
      <div><span class="small">Probabilités (Top 3)</span><br><span style="font-size:.82rem;font-weight:600;color:#2E2A55">{top}</span></div></div>"""),
                unsafe_allow_html=True)

# ───────────────────────────── Pages ────────────────────────────────────────
def page_dashboard(R):
    entete("Prédiction des attaques classiques en cyber",
           "Analyse du trafic &nbsp;•&nbsp; Prédiction des attaques &nbsp;•&nbsp; Évaluation du modèle &nbsp;•&nbsp; Test de scénarios")
    meta, T = R["meta"], R["test"]
    att = meta[meta["pred"] != "Normal"]
    k = st.columns(4)
    k[0].markdown(kpi("🛡️", "#EDE5FF", "#6D28D9", "Trafic analysé", nf(len(meta)), "flux • jeu de test (15 %)"), unsafe_allow_html=True)
    k[1].markdown(kpi("⚠️", "#FDE8E8", "#DC2626", "Attaques prédites", nf(len(att)), f"dont {nf((att['vrai']==att['pred']).sum())} confirmées"), unsafe_allow_html=True)
    k[2].markdown(kpi("🎯", "#FEF3C7", "#D97706", "Taux d’attaque", pc(len(att) / len(meta)), "attaques / trafic analysé"), unsafe_allow_html=True)
    k[3].markdown(kpi("✔️", "#DCFCE7", "#16A34A", "Précision du modèle", pc(T["Accuracy"]), f"accuracy • {R['deploye']}"), unsafe_allow_html=True)
    st.write("")

    gauche, droite = st.columns([2.05, 1], gap="medium")
    with gauche:
        a, b = st.columns([1.35, 1], gap="medium")
        with a.container(border=True):
            titre_carte("📈", "Tendances du trafic réseau (par heure)")
            h = meta.groupby("heure")["packet_count"].sum().reindex(range(24), fill_value=0)
            ha = att.groupby("heure")["packet_count"].sum().reindex(range(24), fill_value=0)
            fig = go.Figure()
            fig.add_scatter(x=[f"{i:02d}h" for i in h.index], y=h.values, name="Trafic total", mode="lines",
                            line=dict(color="#7C3AED", width=2.5, shape="spline"), fill="tozeroy", fillcolor="rgba(124,58,237,.10)")
            fig.add_scatter(x=[f"{i:02d}h" for i in ha.index], y=ha.values, name="Trafic d’attaque (prédit)", mode="lines",
                            line=dict(color="#EF4444", width=2.5, shape="spline"))
            for s, c in ((h, "#7C3AED"), (ha, "#EF4444")):
                fig.add_annotation(x=f"{int(s.idxmax()):02d}h", y=s.max(), text=nf(s.max() / 1000) + " k", showarrow=True,
                                   arrowhead=0, ax=0, ay=-22, bgcolor=c, font=dict(color="#fff", size=10), bordercolor=c)
            fig.update_yaxes(title="Nombre de paquets", gridcolor="#F0EBFF", zeroline=False)
            fig.update_xaxes(gridcolor="rgba(0,0,0,0)", dtick=3)
            fig.update_layout(legend=dict(orientation="h", y=1.18, x=0))
            st.plotly_chart(style_fig(fig, 250, True), width="stretch")
        with b.container(border=True):
            titre_carte("📊", "Performances du modèle")
            c1, c2 = st.columns([1, 1])
            with c1:
                fig = go.Figure(go.Pie(values=[T["Accuracy"], 1 - T["Accuracy"]], hole=0.74, sort=False, textinfo="none",
                                       marker=dict(colors=["#7C3AED", "#EDE5FF"]), hoverinfo="skip"))
                fig.add_annotation(text=f"<b style='font-size:20px;color:#2E1065'>{nf(100*T['Accuracy'],2)} %</b><br><span style='font-size:11px'>Accuracy</span>",
                                   showarrow=False)
                st.plotly_chart(style_fig(fig, 200), width="stretch")
            with c2:
                lig = [("Précision", T["Précision"], "#7C3AED"), ("Rappel (Recall)", T["Recall"], "#A78BFA"),
                       ("Spécificité", T["Spécificité"], "#EC4899"), ("F1-score", T["F1-score"], "#F59E0B"), ("AUC (ROC)", T["AUC"], "#10B981")]
                st.markdown('<div class="leg">' + "".join(
                    f'<div class="row"><span><span class="dt" style="background:{c}"></span>{n}</span><b>{pc(v)}</b></div>'
                    for n, v, c in lig) + "</div>", unsafe_allow_html=True)
        a, b = st.columns([1.1, 1], gap="medium")
        cnt = att["pred"].value_counts()
        with a.container(border=True):
            titre_carte("📊", "Fréquence des attaques par classe")
            c = cnt.sort_values()
            fig = go.Figure(go.Bar(x=c.values, y=[aff(i) for i in c.index], orientation="h", text=[nf(v) for v in c.values],
                                   textposition="outside", marker=dict(color=[COULEURS[i] for i in c.index]), cliponaxis=False))
            fig.update_xaxes(gridcolor="#F0EBFF", title="Nombre d’attaques (prédites)")
            st.plotly_chart(style_fig(fig, 300), width="stretch")
        with b.container(border=True):
            titre_carte("🥧", "Répartition des attaques (10 classes)")
            fig = go.Figure(go.Pie(labels=[aff(i) for i in cnt.index], values=cnt.values, hole=0.55, sort=True,
                                   marker=dict(colors=[COULEURS[i] for i in cnt.index]), textinfo="percent", textfont=dict(size=10, color="#fff")))
            fig.add_annotation(text=f"<b style='font-size:18px;color:#2E1065'>{nf(len(att))}</b><br><span style='font-size:11px'>attaques</span>", showarrow=False)
            fig.update_layout(legend=dict(font=dict(size=10), x=1.0, y=0.5))
            st.plotly_chart(style_fig(fig, 300, True), width="stretch")
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            with c1: titre_carte("🕘", "Incidents récents (prédictions)")
            c2.button("Voir tout →", key="voir_tout", on_click=lambda: st.session_state.update(page="Incidents"))
            rec = att.sort_values("timestamp_utc", ascending=False).head(5)
            st.markdown(table_incidents(rec), unsafe_allow_html=True)
    with droite:
        with st.container(border=True):
            titre_carte("🧪", "Partie de test")
            st.caption("Modifiez les paramètres du trafic et testez la prédiction.")
            panneau_test(R)

def page_test(R):
    entete("Prédiction / Test", "Modifiez les paramètres avec les barres, puis lancez la prédiction (Normal ou l’une des 10 attaques)")
    with st.container(border=True):
        titre_carte("🧪", "Partie de test")
        panneau_test(R, large=True)
    with st.container(border=True):
        titre_carte("📂", "Prédiction par lot (fichier CSV)")
        st.caption("Chargez un CSV avec les mêmes colonnes que le dataset (les colonnes inutiles sont ignorées).")
        ech = charger_csv().sample(100, random_state=1).drop(columns=["attack_type", "is_attack"])
        st.download_button("Télécharger un exemple (100 lignes)", ech.to_csv(index=False).encode("utf-8"), "exemple_trafic.csv", "text/csv")
        f = st.file_uploader("Fichier CSV", type="csv")
        if f is not None:
            d = pd.read_csv(f)
            manque = [c for c in NUM + CATEGORIES if c not in d.columns]
            if manque:
                st.error("Colonnes manquantes : " + ", ".join(manque))
            else:
                X = aligner(R, d.drop(columns=[c for c in COLS_IDENTIFIANTS + ["is_attack"] if c in d.columns]).fillna(0))
                P = predire_proba(R, X)
                out = d.copy()
                out["classe_predite"] = [aff(CLASSES[i]) for i in P.argmax(1)]
                out["confiance"] = P.max(1).round(4)
                out["resultat"] = np.where(P.argmax(1) == 0, "Normal", "Attaque")
                st.metric("Attaques détectées", f"{(P.argmax(1) > 0).sum()} / {len(out)}")
                st.dataframe(out[["classe_predite", "confiance", "resultat"] + [c for c in out.columns[:6]]], width="stretch")
                st.download_button("Télécharger les prédictions", out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

def page_analyses(R):
    entete("Analyses", "Statistiques du jeu de données et variables discriminantes")
    df = charger_csv()
    comp = df["attack_type"].value_counts().reindex(CLASSES)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes", nf(len(df))); c2.metric("Colonnes", df.shape[1])
    c3.metric("Trafic normal", pc((df.is_attack == 0).mean(), 1)); c4.metric("Attaques", pc(df.is_attack.mean(), 1))
    a, b = st.columns(2, gap="medium")
    with a.container(border=True):
        titre_carte("📊", "Déséquilibre des classes")
        fig = go.Figure(go.Bar(x=[aff(c) for c in comp.index], y=comp.values, marker_color=[COULEURS[c] for c in comp.index],
                               text=[nf(v) for v in comp.values], textposition="outside"))
        fig.update_yaxes(gridcolor="#F0EBFF", type="log", title="Lignes (échelle log)")
        st.plotly_chart(style_fig(fig, 320), width="stretch")
        st.caption("Un modèle qui répondrait toujours « Normal » aurait déjà "
                   f"{pc((df.is_attack == 0).mean(), 1)} d’accuracy : on suit donc surtout Précision, Recall et F1.")
    with b.container(border=True):
        titre_carte("🌲", "Variables les plus importantes (Random Forest)")
        imp = R["imp"].head(15).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#7C3AED"))
        fig.update_xaxes(gridcolor="#F0EBFF")
        st.plotly_chart(style_fig(fig, 320), width="stretch")
    with st.container(border=True):
        titre_carte("🔎", "Signature de chaque attaque (écart attaque / normal en σ)")
        st.dataframe(pd.Series({aff(k): v for k, v in R["sig"].items()}, name="Variables les plus discriminantes").to_frame(), width="stretch")
    with st.container(border=True):
        titre_carte("🧮", "Statistiques descriptives")
        st.dataframe(df[NUM].describe().T.round(2), width="stretch")

def page_incidents(R):
    entete("Incidents", "Détections sur le jeu de test et historique de vos tests")
    meta = R["meta"]
    with st.container(border=True):
        titre_carte("🚨", "Incidents détectés (jeu de test)")
        c1, c2 = st.columns([2, 1])
        choix = c1.multiselect("Classes", [aff(c) for c in ATTAQUES], default=[aff(c) for c in ATTAQUES])
        seuil = c2.slider("Confiance minimale", 0.0, 1.0, st.session_state.get("seuil", 0.5), 0.01)
        d = meta[(meta["pred"] != "Normal") & (meta["pred"].map(aff).isin(choix)) & (meta["conf"] >= seuil)]
        d = d.sort_values("timestamp_utc", ascending=False)
        st.caption(f"{nf(len(d))} incidents affichés (les 25 plus récents ci-dessous).")
        st.markdown(table_incidents(d.head(25)), unsafe_allow_html=True)
        st.download_button("Télécharger (CSV)", d.drop(columns="heure").to_csv(index=False).encode("utf-8"), "incidents.csv", "text/csv")
    with st.container(border=True):
        titre_carte("📝", "Journal de vos tests")
        j = st.session_state.journal
        if j:
            jd = pd.DataFrame(j)
            st.dataframe(jd, width="stretch")
            st.download_button("Télécharger le journal (CSV)", jd.to_csv(index=False).encode("utf-8"), "journal_tests.csv", "text/csv")
        else:
            st.info("Aucun test journalisé pour le moment. Lancez une prédiction avec « Journaliser le test » activé.")

def page_donnees(R):
    entete("Données & Modèle", "Chargement, prétraitement et comparaison des algorithmes")
    df, I = charger_csv(), R["info"]
    with st.container(border=True):
        titre_carte("🗂️", "Aperçu du dataset.csv")
        st.dataframe(df.head(10), width="stretch")
        c = st.columns(5)
        c[0].metric("Lignes", nf(I["n_brut"])); c[1].metric("Doublons supprimés", nf(I["n_doublons"]))
        c[2].metric("NaN remplacés par 0", nf(I["n_nan"])); c[3].metric("Variables (après encodage)", I["n_features"])
        c[4].metric("Temps d’entraînement", f"{I['duree']:.0f} s")
    a, b = st.columns(2, gap="medium")
    with a.container(border=True):
        titre_carte("⚙️", "Pipeline (identique au notebook)")
        st.markdown(f"""
1. Suppression des **doublons** (hors `sample_id`) → {nf(I['n_final'])} lignes
2. Valeurs manquantes → `fillna(0)`
3. Cible : `attack_type` (Normal + 10 attaques) — exclus des variables : `sample_id`, `timestamp_utc`, IP, `src_port`, `attack_type`, `is_attack`
4. `get_dummies` sur `protocol`, `service`, `dst_port`
5. Découpage stratifié **70 / 15 / 15** : {nf(I['n_train'])} / {nf(I['n_val'])} / {nf(I['n_test'])}
6. `StandardScaler` ajusté sur le train uniquement
7. 5 modèles comparés sur la validation, évaluation finale sur le test
""")
    with b.container(border=True):
        titre_carte("🏆", "Modèle utilisé pour la prédiction")
        st.success(f"**{R['deploye']}** (meilleur F1 de validation parmi les modèles qui fournissent des probabilités)")
        if R["meilleur_global"] != R["deploye"]:
            st.caption(f"Meilleur F1 global : {R['meilleur_global']} (ne fournit pas de probabilités).")
        st.dataframe(R["tab_val"].style.format("{:.4f}", subset=["Accuracy", "Précision", "Recall", "F1-score"]).format("{:.1f}", subset=["Temps (s)"]),
                     width="stretch")
        st.caption("Métriques sur la validation, moyenne macro des 11 classes. La régression linéaire n’est qu’une référence pédagogique.")

def page_evaluation(R):
    entete("Évaluation", f"Résultats finaux sur le jeu de test — {R['deploye']}")
    T = R["test"]
    c = st.columns(4)
    for col, n in zip(c, ["Accuracy", "Précision", "Recall", "F1-score"]):
        col.markdown(kpi("📐", "#EDE5FF", "#6D28D9", n, pc(T[n]), "test • moyenne macro" if n != "Accuracy" else "test • global"), unsafe_allow_html=True)
    st.write("")
    a, b = st.columns([1.2, 1], gap="medium")
    with a.container(border=True):
        titre_carte("🧩", "Matrice de confusion (test)")
        cm = R["cm"]; noms = [aff(x) for x in CLASSES]
        fig = go.Figure(go.Heatmap(z=cm, x=noms, y=noms, colorscale=[[0, "#FFFFFF"], [1, "#5B21B6"]], text=cm, texttemplate="%{text}",
                                   showscale=False, hovertemplate="Vrai : %{y}<br>Prédit : %{x}<br>%{z}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", title="Classe réelle"); fig.update_xaxes(title="Classe prédite", tickangle=-35)
        st.plotly_chart(style_fig(fig, 470), width="stretch")
    with b.container(border=True):
        titre_carte("📋", "Détail par classe")
        r = R["rapport"].rename(columns={"precision": "Précision", "recall": "Recall", "f1-score": "F1-score", "support": "Effectif"})
        st.dataframe(r.style.format({"Précision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}", "Effectif": "{:.0f}"}), width="stretch", height=430)
    with st.container(border=True):
        titre_carte("🛡️", "Détection binaire : Attaque vs Normal")
        d = st.columns(5)
        d[0].metric("Précision", pc(T["Détection (attaque vs normal) — Précision"]))
        d[1].metric("Recall", pc(T["Détection (attaque vs normal) — Recall"]))
        d[2].metric("F1-score", pc(T["Détection (attaque vs normal) — F1"]))
        d[3].metric("Spécificité", pc(T["Spécificité"])); d[4].metric("AUC (ROC)", pc(T["AUC"]))
        st.caption("Faux négatif = attaque non détectée ; faux positif = trafic normal signalé à tort. "
                   "Données simulées : ces scores ne prouvent pas une efficacité en production.")

def page_parametres(R):
    entete("Paramètres", "Compte, test et maintenance")
    a, b = st.columns(2, gap="medium")
    with a.container(border=True):
        titre_carte("👤", "Compte")
        st.write(f"**Nom :** {NOM_COMPLET}"); st.write(f"**Utilisateur :** {UTILISATEUR}")
        if st.button("Se déconnecter", type="primary"):
            st.session_state.clear(); st.rerun()
    with b.container(border=True):
        titre_carte("🔧", "Test & modèle")
        st.session_state["seuil"] = st.slider("Confiance minimale par défaut (page Incidents)", 0.0, 1.0, st.session_state.get("seuil", 0.5), 0.01)
        if st.button("Réinitialiser les paramètres de test"):
            for k in [k for k in st.session_state if k.startswith("w_")] + ["P", "dernier"]:
                st.session_state.pop(k, None)
            st.rerun()
        if st.button("Ré-entraîner les modèles"):
            if os.path.exists(CACHE):
                os.remove(CACHE)
            st.cache_resource.clear(); st.cache_data.clear(); st.rerun()

# ───────────────────────────── Connexion ────────────────────────────────────
def page_login():
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.write(""); st.write("")
        with st.container(border=True):
            st.markdown(f"""<div class="login-card">{logo(78, 'l')}<h2>{NOM_COMPLET}</h2>
            <div class="small">Research • Innovation • Education</div>
            <div style="margin-top:.9rem;font-weight:700;color:#4C1D95">Prédiction des attaques classiques en cyber</div></div>""",
                        unsafe_allow_html=True)
            with st.form("login"):
                u = st.text_input("Nom d’utilisateur")
                p = st.text_input("Mot de passe", type="password")
                ok = st.form_submit_button("Se connecter", width="stretch")
            if ok:
                if hmac.compare_digest(u.strip(), UTILISATEUR) and hmac.compare_digest(p, MOT_DE_PASSE):
                    st.session_state.auth = True; st.rerun()
                else:
                    st.error("Nom d’utilisateur ou mot de passe incorrect.")

# ───────────────────────────── Programme principal ──────────────────────────
st.set_page_config(page_title="Prédiction des attaques cyber", page_icon="🛡️", layout="wide")
injecter_css()

if not st.session_state.get("auth"):
    page_login()
    st.stop()

if not os.path.exists(CSV):
    st.error("Fichier dataset.csv introuvable : placez-le dans le même dossier que app.py.")
    st.stop()

with st.spinner("Chargement du dataset et entraînement des modèles (première exécution uniquement, ≈ 1 à 3 min)…"):
    R = obtenir_modeles()
init_etat(R)

with st.sidebar:
    st.markdown(f'<div class="brand">{logo(46, "s")}<div><div class="t1">{NOM_COMPLET}</div>'
                f'<div class="t2">Research • Innovation • Education</div></div></div>', unsafe_allow_html=True)
    icones = dict(PAGES)
    page = st.radio("Navigation", [p for p, _ in PAGES], key="page", label_visibility="collapsed",
                    format_func=lambda p: f":material/{icones[p]}: {p}")
    st.markdown("<div style='height:40vh'></div>", unsafe_allow_html=True)
    st.caption(f"Connectée : **{UTILISATEUR}**")

{"Tableau de bord": page_dashboard, "Prédiction / Test": page_test, "Analyses": page_analyses,
 "Incidents": page_incidents, "Données & Modèle": page_donnees, "Évaluation": page_evaluation,
 "Paramètres": page_parametres}[page](R)
