#!/usr/bin/env bash
# Kullanım: PAROLA=... paketle.sh GIZLI_KLASOR HEDEF.7z
# Klasörü AES-256 ile şifreler (dosya adları dahil: -mhe=on) ve doğrular. Herhangi bir denetim başarısızsa
# hedef silinir ve 1 ile çıkılır. Parola yalnızca ortam değişkeninden okunur ve hiçbir yere yazdırılmaz.
set -euo pipefail
GIZLI="$1"; HEDEF="$2"
[ -n "${PAROLA:-}" ] || { echo "HATA: PAROLA yok" >&2; exit 1; }
hata() { rm -f "$HEDEF"; echo "HATA: $1; paket silindi." >&2; exit 1; }
rm -f "$HEDEF"
( cd "$GIZLI" && 7z a -t7z -mhe=on -mx=5 -p"$PAROLA" "$HEDEF" . > /dev/null ) || hata "sıkıştırma/şifreleme başarısız"
7z t -p"$PAROLA" "$HEDEF" > /dev/null 2>&1 || hata "bütünlük denetimi başarısız"
# Yanlış parolayla dosya listesi bile okunamamalı (başlık şifrelemesi)
if 7z l -p"yanlis-parola-denetimi" "$HEDEF" > /dev/null 2>&1; then hata "başlıklar şifrelenmemiş"; fi
BEKLENEN=$(cd "$GIZLI" && find . -type f | sed 's|^\./||' | sort)
ICERIK=$(7z l -slt -p"$PAROLA" "$HEDEF" | sed -n 's/^Path = //p' | tail -n +2 | while read -r y; do
  [ -f "$GIZLI/$y" ] && echo "$y"; done | sort)
[ "$BEKLENEN" = "$ICERIK" ] || hata "pakette eksik dosya var"
echo "Şifreli paket hazır: $(basename "$HEDEF") ($(echo "$BEKLENEN" | wc -l) dosya)"