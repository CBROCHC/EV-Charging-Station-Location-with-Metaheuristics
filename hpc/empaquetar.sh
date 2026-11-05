#!/bin/bash
# Uso: bash empaquetar.sh <etiqueta>
# Empaqueta resultados + informe en un .tar.gz y lo copia a un sitio seguro.
# $HOME no tiene copias de seguridad; $HOME/W es la carpeta personal montada como unidad W.
ETIQUETA=${1:-X}
cd "$HOME/metaheuristicas" || exit 1
CARPS=""
for c in resultados informe prueba informe_prueba; do
    [ -d "$c" ] && CARPS="$CARPS $c"
done
PAQUETE="paquete_${ETIQUETA}_${SLURM_JOB_ID:-manual}.tar.gz"
tar czf "$PAQUETE" $CARPS
echo "Paquete: $PWD/$PAQUETE ($(du -h "$PAQUETE" | cut -f1))"
if [ -d "$HOME/W" ]; then
    mkdir -p "$HOME/W/metaheuristicas" \
      && cp "$PAQUETE" "$HOME/W/metaheuristicas/" \
      && echo "Copiado a $HOME/W/metaheuristicas/" \
      || echo "AVISO: no se pudo copiar a W"
else
    echo "AVISO: $HOME/W no existe en este nodo; el paquete queda solo en $PWD"
fi
