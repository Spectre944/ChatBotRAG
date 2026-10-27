// Глобальная функция открытия окна
window.openChatModal = function (e) {
  if (e && e.preventDefault) e.preventDefault();
  const chatWindow = document.getElementById('ecotest-chat-window');
  if (chatWindow) {
    chatWindow.style.display = 'flex';
  }
};

const CHAT_SCRIPT = document.currentScript;
const DEVICE = CHAT_SCRIPT?.dataset.device || null;

const SESSION_ID = (() => {
  try {
    let id = sessionStorage.getItem('ecotest_sid');
    if (!id) { id = crypto.randomUUID(); sessionStorage.setItem('ecotest_sid', id); }
    return id;
  } catch { return crypto.randomUUID(); }
})();

document.addEventListener('DOMContentLoaded', function () {
  // 0. Подключаем внешнюю библиотеку marked.js для рендеринга Markdown
  if (!window.marked) {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
    document.head.appendChild(script);
  }

  // 1. Внедрение CSS стилей
  const styles = `
    #ecotest-chat-trigger {
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 99999;
      background-color: #0B7865;
      color: #ffffff;
      border: none;
      border-radius: 50px;
      padding: 14px 24px;
      font-family: 'Onest', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 15px;
      font-weight: 600;
      box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 10px;
      transition: transform 0.2s ease, background-color 0.2s ease;
    }
    #ecotest-chat-trigger:hover {
      background-color: #085a4c;
      transform: translateY(-2px);
    }

    #ecotest-chat-window {
      position: fixed;
      bottom: 24px;
      right: 24px;
      width: 650px;
      max-width: calc(100vw - 48px);
      height: 700px;
      max-height: calc(100vh - 48px);
      background-color: #ffffff;
      border-radius: 16px;
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
      z-index: 99999;
      display: none;
      flex-direction: column;
      overflow: hidden;
      font-family: 'Onest', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      border: 1px solid #e5e7eb;
    }

    .ecotest-chat-header {
      background-color: #0B7865;
      color: #ffffff;
      padding: 16px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .ecotest-chat-header-title {
      font-size: 17px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .ecotest-chat-close {
      background: transparent;
      border: none;
      color: #ffffff;
      font-size: 24px;
      cursor: pointer;
      line-height: 1;
      opacity: 0.8;
      transition: opacity 0.2s;
    }
    .ecotest-chat-close:hover {
      opacity: 1;
    }

    .ecotest-chat-messages {
      flex: 1;
      padding: 20px;
      overflow-y: auto;
      background-color: #f9fafb;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .ecotest-message-wrapper {
      display: flex;
      flex-direction: column;
      max-width: 85%;
    }
    .ecotest-message-wrapper.bot {
      align-self: flex-start;
    }
    .ecotest-message-wrapper.user {
      align-self: flex-end;
    }

    /* Стилизацию сообщений и Markdown-списков */
    .ecotest-message {
      padding: 12px 18px;
      border-radius: 14px;
      font-size: 15px;
      line-height: 1.6;
      word-break: break-word;
    }
    .ecotest-message p {
      margin: 0 0 8px 0;
    }
    .ecotest-message p:last-child {
      margin-bottom: 0;
    }
    .ecotest-message ol, .ecotest-message ul {
      margin: 6px 0 6px 20px;
      padding: 0;
    }
    .ecotest-message li {
      margin-bottom: 4px;
    }

    .ecotest-message-wrapper.bot .ecotest-message {
      background-color: #ffffff;
      color: #1f2937;
      border: 1px solid #e5e7eb;
      border-bottom-left-radius: 4px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .ecotest-message-wrapper.user .ecotest-message {
      background-color: #0B7865;
      color: #ffffff;
      border-bottom-right-radius: 4px;
    }

    .ecotest-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 6px;
      padding-left: 4px;
    }
    .ecotest-action-btn {
      background: transparent;
      border: none;
      cursor: pointer;
      font-size: 14px;
      color: #6b7280;
      padding: 4px 6px;
      border-radius: 6px;
      transition: background-color 0.2s, color 0.2s;
      display: flex;
      align-items: center;
      gap: 4px;
    }
    .ecotest-action-btn:hover {
      background-color: #e5e7eb;
      color: #111827;
    }
    .ecotest-action-btn.active-like {
      color: #10b981;
      font-weight: bold;
    }
    .ecotest-action-btn.active-dislike {
      color: #ef4444;
      font-weight: bold;
    }

    .ecotest-chat-input-area {
      display: flex;
      padding: 14px 16px;
      background-color: #ffffff;
      border-top: 1px solid #e5e7eb;
      align-items: center;
      gap: 10px;
    }
    .ecotest-chat-input {
      flex: 1;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 15px;
      outline: none;
      transition: border-color 0.2s;
    }
    .ecotest-chat-input:focus {
      border-color: #0B7865;
    }
    .ecotest-chat-send {
      background-color: #0B7865;
      color: #ffffff;
      border: none;
      border-radius: 8px;
      padding: 10px 18px;
      cursor: pointer;
      font-size: 15px;
      font-weight: 600;
      transition: background-color 0.2s;
    }
    .ecotest-chat-send:hover {
      background-color: #085a4c;
    }
    .ecotest-chat-send:disabled {
      background-color: #9ca3af;
      cursor: not-allowed;
    }
  `;

  const styleSheet = document.createElement("style");
  styleSheet.type = "text/css";
  styleSheet.innerText = styles;
  document.head.appendChild(styleSheet);

  // 2. Внедрение HTML-структуры
  const widgetContainer = document.createElement("div");
  widgetContainer.id = "ecotest-widget-root";
  widgetContainer.innerHTML = `
    <button id="ecotest-chat-trigger" aria-label="Відкрити чат підтримки">
      💬 Онлайн-консультант
    </button>
    <div id="ecotest-chat-window">
      <div class="ecotest-chat-header">
        <div class="ecotest-chat-header-title">🤖 ECOTEST AI Асистент</div>
        <button class="ecotest-chat-close" id="ecotest-chat-close" aria-label="Закрити чат">&times;</button>
      </div>
      <div class="ecotest-chat-messages" id="ecotest-chat-messages">
        <div class="ecotest-message-wrapper bot">
          <div class="ecotest-message">Вітаємо! 👋 Як я можу вам допомогти щодо обладнання ECOTEST?</div>
        </div>
      </div>
      <div class="ecotest-chat-input-area">
        <input type="text" id="ecotest-chat-input" class="ecotest-chat-input" placeholder="Поставте запитання про прилад..." />
        <button id="ecotest-chat-send" class="ecotest-chat-send">Надіслати</button>
      </div>
    </div>
  `;
  document.body.appendChild(widgetContainer);

  // 3. Элементы и логика
  const trigger = document.getElementById('ecotest-chat-trigger');
  const chatWindow = document.getElementById('ecotest-chat-window');
  const closeBtn = document.getElementById('ecotest-chat-close');
  const sendBtn = document.getElementById('ecotest-chat-send');
  const input = document.getElementById('ecotest-chat-input');
  const messages = document.getElementById('ecotest-chat-messages');

  trigger.addEventListener('click', () => {
    chatWindow.style.display = chatWindow.style.display === 'flex' ? 'none' : 'flex';
  });

  closeBtn.addEventListener('click', () => {
    chatWindow.style.display = 'none';
  });

  function appendMessageWrapper(sender) {
    const wrapper = document.createElement('div');
    wrapper.className = `ecotest-message-wrapper ${sender}`;

    const msgDiv = document.createElement('div');
    msgDiv.className = 'ecotest-message';
    wrapper.appendChild(msgDiv);

    messages.appendChild(wrapper);
    messages.scrollTop = messages.scrollHeight;
    return { wrapper, msgDiv };
  }

  function attachActions(wrapper, msgDiv) {
    const actionsDiv = document.createElement('div');
    actionsDiv.className = 'ecotest-actions';

    const likeBtn = document.createElement('button');
    likeBtn.className = 'ecotest-action-btn';
    likeBtn.innerHTML = '👍';
    likeBtn.title = 'Корисна відповідь';

    const dislikeBtn = document.createElement('button');
    dislikeBtn.className = 'ecotest-action-btn';
    dislikeBtn.innerHTML = '👎';
    dislikeBtn.title = 'Некоректна відповідь';

    const copyBtn = document.createElement('button');
    copyBtn.className = 'ecotest-action-btn';
    copyBtn.innerHTML = '📋';
    copyBtn.title = 'Скопіювати';

    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(msgDiv.innerText);
      copyBtn.innerHTML = '✅';
      setTimeout(() => copyBtn.innerHTML = '📋', 2000);
    });

	likeBtn.addEventListener('click', () => {
		const on = likeBtn.classList.toggle('active-like');
		dislikeBtn.classList.remove('active-dislike');
		if (on) sendFeedback(messageId, 1);
	});
	dislikeBtn.addEventListener('click', () => {
		const on = dislikeBtn.classList.toggle('active-dislike');
		likeBtn.classList.remove('active-like');
		if (on) sendFeedback(messageId, -1);
	});

    actionsDiv.appendChild(likeBtn);
    actionsDiv.appendChild(dislikeBtn);
    actionsDiv.appendChild(copyBtn);

    wrapper.appendChild(actionsDiv);
    messages.scrollTop = messages.scrollHeight;
  }

	async function sendFeedback(messageId, rating, comment = null) {
	  if (!messageId) return;
	  try {
		await fetch('/rate', {
		  method: 'POST',
		  headers: { 'Content-Type': 'application/json' },
		  body: JSON.stringify({ message_id: messageId, session_id: SESSION_ID, rating, comment })
		});
	  } catch (e) {
		console.error('Не удалось отправить оценку:', e);
	  }
	}

// Регулируем переносы и списки
  function formatMarkdownText(text) {
    return text
      // Восстанавливаем оригинальные переносы строк
      .replace(/\\n/g, '\n')
      // Авто-перенос перед цифрой списка (1., 2., 3.), если текст слепился в одну строку
      .replace(/([^\n])(\d+\.\s)/g, '$1\n$2');
  }

  // Отрисовка с помощью Marked.js
  function renderMarkdown(element, rawText) {
    const formattedText = formatMarkdownText(rawText);
    if (window.marked) {
      element.innerHTML = window.marked.parse(formattedText);
    } else {
      element.textContent = formattedText;
    }
  }

  async function sendMessage() {
    const text = input.value.trim();
    if (!text) return;

    const userObj = appendMessageWrapper('user');
    userObj.msgDiv.textContent = text;

    input.value = '';
    sendBtn.disabled = true;

    const botObj = appendMessageWrapper('bot');
    let rawAccumulatedText = '';

    try {
      
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text, device: DEVICE, session_id: SESSION_ID })
      });

      if (!response.ok) throw new Error(`Помилка сервера: ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop();

		for (const line of lines) {
		  if (!line.startsWith('data: ')) continue;
		  const rawData = line.slice(6);          // без trim, иначе теряются пробелы в конце токенов
		  if (rawData === '[DONE]') break;
		  if (rawData.startsWith('[ID:')) {       // служебный пакет с ID ответа
			messageId = rawData.slice(4, -1);
			continue;
		  }
		  rawAccumulatedText += rawData;
		  renderMarkdown(botObj.msgDiv, rawAccumulatedText);
		  messages.scrollTop = messages.scrollHeight;
		}
      }

      attachActions(botObj.wrapper, botObj.msgDiv, messageId);

    } catch (error) {
      console.error('Chatbot Stream Error:', error);
      botObj.msgDiv.textContent = 'Вибачте, виникла помилка під час з\'єднання з сервером.';
    } finally {
      sendBtn.disabled = false;
      messages.scrollTop = messages.scrollHeight;
    }
  }

  sendBtn.addEventListener('click', sendMessage);
  input.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
  });
});