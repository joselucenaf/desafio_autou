document.addEventListener('DOMContentLoaded', () => {
    const textTabBtn = document.getElementById('textTabBtn');
    const fileTabBtn = document.getElementById('fileTabBtn');
    const textPane = document.getElementById('textPane');
    const filePane = document.getElementById('filePane');
    const fileInput = document.getElementById('fileInput');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultArea = document.getElementById('resultArea');
    const emailInput = document.getElementById('emailInput');


    const switchTab = (mode) => {
        if (mode === 'text') {
            textTabBtn.classList.add('active');
            fileTabBtn.classList.remove('active');
            textPane.classList.remove('hidden');
            filePane.classList.add('hidden');
        } else {
            fileTabBtn.classList.add('active');
            textTabBtn.classList.remove('active');
            filePane.classList.remove('hidden');
            textPane.classList.add('hidden');
        }
    };

    textTabBtn.addEventListener('click', () => switchTab('text'));
    fileTabBtn.addEventListener('click', () => switchTab('file'));

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            fileNameDisplay.innerText = `Selecionado: ${fileInput.files[0].name}`;
        }
    });

    analyzeBtn.addEventListener('click', async () => {
        const text = emailInput.value;
        const isFileMode = !filePane.classList.contains('hidden');
 
        if (isFileMode && fileInput.files.length === 0) {
            alert("Por favor, selecione um arquivo .pdf ou .txt.");
            return;
        }
        if (!isFileMode && !text.trim()) {
            alert("Por favor, insira um texto para análise.");
            return;
        }

        analyzeBtn.innerText = "Processando...";
        analyzeBtn.disabled = true;

        try {
            let response;
            
            if (isFileMode) {
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                response = await fetch('http://127.0.0.1:8000/analyze-file', {
                    method: 'POST',
                    body: formData
                });
            } else {
                response = await fetch('http://127.0.0.1:8000/analyze-email', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email: text })
                });
            }

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Erro no processamento");
            }

            const data = await response.json();
            resultArea.classList.remove('hidden');
            const badge = document.getElementById('categoryBadge');
            
            badge.innerText = data.categoria;
            document.getElementById('suggestedResponse').innerText = data.resposta_sugerida;

            if (data.categoria === "Produtivo") {
                badge.style.background = "rgba(34, 197, 94, 0.2)";
                badge.style.color = "#4ade80";
            } else {
                badge.style.background = "rgba(139, 92, 246, 0.2)";
                badge.style.color = "#a78bfa";
            }

            resultArea.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            alert(`Erro: ${error.message}. Verifique se o backend está ativo.`);
            console.error("Erro na integração:", error);
        } finally {
            analyzeBtn.innerText = "Analisar Conteúdo";
            analyzeBtn.disabled = false;
        }
    });
});