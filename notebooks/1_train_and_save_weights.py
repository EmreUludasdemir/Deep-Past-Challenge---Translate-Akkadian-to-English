"""
==========================================================================
NOTEBOOK 1: EGITIM (Internet ON, GPU ON)
==========================================================================
Bu notebooku Kaggle'da Internet ACIK olarak calistirin.
Egitim bitince model agirliklarini Kaggle Dataset olarak kaydeder.

Kaggle Notebook Ayarlari:
  - Accelerator: GPU T4 x2 (veya P100)
  - Internet: ON
  - Competition Data: deep-past-initiative-machine-translation (otomatik eklenir)

Kullanim:
  Tum hucreleri sirayla calistirin veya tek script olarak:
  !python 1_train_and_save_weights.py --track top_lb
==========================================================================
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

# ---- Hucre 1: Bagimliliklari kur ----
def install_deps():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "sacrebleu", "accelerate", "datasets", "pyyaml",
                           "scikit-learn"])
    print("Bagimliliklar kuruldu.")


# ---- Hucre 2: Kodu working dizinine kopyala ----
def setup_workspace(code_dataset_dir=None):
    """Eger kod bir Kaggle Dataset olarak eklendiyse kopyala."""
    if code_dataset_dir and os.path.isdir(code_dataset_dir):
        for item in os.listdir(code_dataset_dir):
            src = os.path.join(code_dataset_dir, item)
            dst = os.path.join("/kaggle/working", item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        print(f"Kod {code_dataset_dir} -> /kaggle/working kopyalandi.")
    os.chdir("/kaggle/working")
    print(f"Calisma dizini: {os.getcwd()}")


# ---- Hucre 3: Egitimi baslat ----
def run_training(track="top_lb", smoke=False):
    from utils import load_yaml
    from train import run_training as _train

    config_path = f"configs/{track}.yaml"
    config = load_yaml(config_path)

    # Kaggle path'lerini ayarla
    config["global"]["data_dir"] = "/kaggle/input/deep-past-initiative-machine-translation"
    config["global"]["output_dir"] = f"/kaggle/working/outputs/{track}"

    if smoke:
        config.setdefault("debug", {})["smoke_max_rows"] = 50

    os.makedirs(config["global"]["output_dir"], exist_ok=True)
    print(f"=== EGITIM BASLIYOR: {track} ===")
    t0 = time.time()

    summary = _train(config)

    elapsed = time.time() - t0
    print(f"Egitim tamamlandi: {elapsed:.0f} saniye")
    print(f"Summary: {json.dumps(summary, indent=2, default=str)}")
    return summary


# ---- Hucre 4: Agirliklari Kaggle Dataset olarak kaydet ----
def save_weights_as_dataset(track="top_lb", dataset_slug="akkadian-weights"):
    """
    Egitilmis model checkpoint'larini /kaggle/working/weights_upload/ altina
    toplar. Kaggle notebook'ta 'Save Version' ile Dataset olarak kaydedebilirsiniz.

    Alternatif: Kaggle API ile otomatik upload.
    """
    output_dir = f"/kaggle/working/outputs/{track}"
    weights_dir = "/kaggle/working/weights_upload"
    os.makedirs(weights_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "training_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"training_summary.json bulunamadi: {summary_path}")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Her modelin en iyi checkpoint'ini kopyala
    for model_info in summary.get("models", []):
        model_name = model_info["name"]
        stages = model_info.get("stages", [])
        if not stages:
            continue

        # En iyi stage'i sec (en yuksek val_geo_mean)
        best_stage = max(stages, key=lambda s: s.get("val_geo_mean", -1.0))
        ckpt_dir = best_stage["checkpoint_dir"]

        if not os.path.isdir(ckpt_dir):
            print(f"UYARI: Checkpoint bulunamadi: {ckpt_dir}")
            continue

        dest = os.path.join(weights_dir, model_name)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(ckpt_dir, dest)
        print(f"  {model_name}: {ckpt_dir} -> {dest}")
        print(f"    val_geo_mean = {best_stage.get('val_geo_mean', 'N/A')}")

    # Summary'yi de kopyala
    shutil.copy2(summary_path, os.path.join(weights_dir, "training_summary.json"))

    # Dosya listesini goster
    print(f"\nWeights dizini: {weights_dir}")
    for root, dirs, files in os.walk(weights_dir):
        level = root.replace(weights_dir, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 2 * (level + 1)
        for f in files:
            fpath = os.path.join(root, f)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"{sub_indent}{f} ({size_mb:.1f} MB)")

    print("\n" + "=" * 60)
    print("SONRAKI ADIM:")
    print("=" * 60)
    print("1) Bu notebook'u 'Save Version' ile kaydedin")
    print("   - Save output: 'Always save output when creating a Quick Save'")
    print("2) Notebook output'undan 'New Dataset' olusturun:")
    print(f"   - Kaggle > Your Work > Bu Notebook > Output > 'New Dataset'")
    print(f"   - Dataset adi: {dataset_slug}")
    print("3) Submission notebook'unda bu dataset'i ekleyin:")
    print(f"   - /kaggle/input/{dataset_slug}/")
    print("=" * 60)

    return weights_dir


# ---- Kaggle API ile otomatik upload (opsiyonel) ----
def upload_weights_api(weights_dir, dataset_id, message="Model weights"):
    """Kaggle API kurulu ve auth varsa otomatik upload."""
    metadata = {
        "id": dataset_id,
        "title": dataset_id.split("/")[-1],
        "licenses": [{"name": "CC0-1.0"}],
    }
    metadata_path = os.path.join(weights_dir, "dataset-metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    try:
        subprocess.run(
            ["kaggle", "datasets", "version", "-p", weights_dir, "-m", message, "-r", "zip"],
            check=True,
        )
        print(f"Weights yuklendi: {dataset_id}")
    except FileNotFoundError:
        print("Kaggle CLI bulunamadi. Manuel upload yapin (yukaridaki adimlari takip edin).")
    except subprocess.CalledProcessError as e:
        print(f"Upload hatasi: {e}")
        print("Manuel upload yapin.")


def main(
    track="top_lb",
    smoke=False,
    code_dataset=None,
    dataset_slug="akkadian-weights",
    skip_install=False,
):
    """
    Notebook icinden dogrudan cagirin:
        main(track="top_lb")
    veya terminalden:
        python 1_train_and_save_weights.py --track top_lb
    """
    if not skip_install:
        install_deps()

    if os.path.exists("/kaggle/working"):
        setup_workspace(code_dataset)

    summary = run_training(track=track, smoke=smoke)
    weights_dir = save_weights_as_dataset(track=track, dataset_slug=dataset_slug)

    print("\nEGITIM TAMAMLANDI!")


if __name__ == "__main__":
    # Notebook ortaminda argparse sorun cikarir, try/except ile handle et
    try:
        parser = argparse.ArgumentParser(description="Egitim + Weights Kaydetme")
        parser.add_argument("--track", default="top_lb", choices=["baseline", "top_lb"])
        parser.add_argument("--smoke", action="store_true")
        parser.add_argument("--code-dataset", default=None)
        parser.add_argument("--dataset-slug", default="akkadian-weights")
        parser.add_argument("--skip-install", action="store_true")
        args, _ = parser.parse_known_args()  # Bilinmeyen argumanlari yoksay
        main(
            track=args.track,
            smoke=args.smoke,
            code_dataset=args.code_dataset,
            dataset_slug=args.dataset_slug,
            skip_install=args.skip_install,
        )
    except SystemExit:
        # Notebook'da argparse hata verirse varsayilanlarla calistir
        main()
