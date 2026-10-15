/**
 * NeumaPredict AI - Frontend Interaction & Ingestion Client
 * Tecnicatura Superior en Desarrollo de Software - ISPC
 * Paleta del Proyecto: #5269E4 (Azul Real), #E3D252 (Oro), #E3AF52 (Ocre), #565863 (Pizarra)
 */

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('prediction-form');
  const rimInput = document.getElementById('input-rim');
  const widthInput = document.getElementById('input-width');
  const ratioInput = document.getElementById('input-ratio');
  const brandSelect = document.getElementById('select-brand');
  const chkAutoRatio = document.getElementById('chk-auto-ratio');
  const ratioHint = document.getElementById('ratio-hint');

  const btnSubmit = document.getElementById('btn-submit');
  const btnReset = document.getElementById('btn-reset');
  const resultPrice = document.getElementById('result-price');
  const resultSpecsSummary = document.getElementById('result-specs-summary');

  // Control del checkbox de auto-imputación de perfil
  chkAutoRatio.addEventListener('change', () => {
    if (chkAutoRatio.checked) {
      ratioInput.disabled = true;
      ratioInput.dataset.previousValue = ratioInput.value;
      ratioInput.value = '';
      ratioInput.placeholder = 'Imputación mediana (automática)';
      ratioHint.textContent = 'El preprocesador imputa la mediana aprendida en entrenamiento (sin fuga de datos).';
      ratioHint.style.color = 'var(--color-ochre)';
    } else {
      ratioInput.disabled = false;
      ratioInput.value = ratioInput.dataset.previousValue || '55.0';
      ratioInput.placeholder = 'ej. 55';
      ratioHint.textContent = 'Relación porcentual de la altura del flanco.';
      ratioHint.style.color = '';
    }
  });

  // Manejo de los 5 Presets de Prueba Rápida
  document.querySelectorAll('.preset-card, .preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const rim = btn.dataset.rim;
      const width = btn.dataset.width;
      const ratio = btn.dataset.ratio;
      const brand = btn.dataset.brand;

      rimInput.value = rim;
      widthInput.value = width;
      brandSelect.value = brand;

      if (!ratio || ratio === '') {
        chkAutoRatio.checked = true;
        ratioInput.disabled = true;
        ratioInput.value = '';
        ratioInput.placeholder = 'Imputación mediana';
        ratioHint.textContent = 'Segmento pesado: perfil ausente, resuelto por preprocesador.';
        ratioHint.style.color = 'var(--color-ochre)';
      } else {
        chkAutoRatio.checked = false;
        ratioInput.disabled = false;
        ratioInput.value = ratio;
        ratioHint.textContent = 'Relación porcentual de la altura del flanco.';
        ratioHint.style.color = '';
      }

      // Enviar inmediatamente para feedback instantáneo
      executePrediction();
    });
  });

  // Envío del Formulario
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    executePrediction();
  });

  // Restablecer formulario
  btnReset.addEventListener('click', () => {
    rimInput.value = '16.0';
    widthInput.value = '205.0';
    ratioInput.disabled = false;
    ratioInput.value = '55.0';
    chkAutoRatio.checked = false;
    brandSelect.value = 'BRAND_A';
    ratioHint.textContent = 'Relación porcentual de la altura del flanco.';
    ratioHint.style.color = '';
    executePrediction();
  });

  // Función asíncrona de predicción
  async function executePrediction() {
    const rim = parseFloat(rimInput.value);
    const width = parseFloat(widthInput.value);
    const isAutoRatio = chkAutoRatio.checked;
    const ratio = isAutoRatio || ratioInput.value === '' ? null : parseFloat(ratioInput.value);
    const brand = brandSelect.value;

    if (isNaN(rim) || isNaN(width)) {
      alert('Por favor complete las dimensiones requeridas (Rin y Ancho).');
      return;
    }

    // Estado visual de carga
    btnSubmit.disabled = true;
    btnSubmit.querySelector('.btn-text').textContent = 'Calculando...';
    resultPrice.style.opacity = '0.35';

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rim: rim,
          width: width,
          ratio: ratio,
          brand: brand
        })
      });

      const data = await response.json();

      if (response.ok && data.status === 'success') {
        const formatted = data.formatted_cost;
        resultPrice.textContent = formatted;
        resultPrice.style.color = 'var(--color-gold)';

        const ratioText = ratio !== null ? `/${ratio}` : ' (perfil imputado)';
        resultSpecsSummary.textContent = `Neumático ${width}${ratioText} R${rim} · Fabricante ${brand}`;
      } else {
        resultPrice.textContent = 'Error';
        resultPrice.style.color = '#ef4444';
        resultSpecsSummary.textContent = data.message || 'No se pudo generar la predicción.';
      }
    } catch (err) {
      resultPrice.textContent = 'Offline';
      resultPrice.style.color = 'var(--color-ochre)';
      resultSpecsSummary.textContent = 'Servidor local no disponible. Verifique que `py demo_app/server.py` esté activo.';
    } finally {
      btnSubmit.disabled = false;
      btnSubmit.querySelector('.btn-text').textContent = 'Calcular Costo Estimado';
      resultPrice.style.opacity = '1';
    }
  }

  // Inferencia inicial al cargar
  executePrediction();
});
