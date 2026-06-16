document.addEventListener("DOMContentLoaded", function () {
    const chatForm     = document.getElementById("chat-form");
    const userInput    = document.getElementById("user-input");
    const chatMessages = document.getElementById("chat-messages");
    const historyList  = document.querySelector(".history-list");
    const newChatBtn   = document.querySelector(".new-chat-btn");

    let SESSION_ID = "user_" + Math.random().toString(36).slice(2, 8);
    let isLoading    = false;

    // Khởi tạo trạng thái ban đầu (Xóa sạch history fake, đặt lại khung chat)
    resetChatWorkspace();

    function resetChatWorkspace() {
        chatMessages.innerHTML = `
            <div class="message ai">
                <div class="avatar"><i class="fa-solid fa-robot"></i></div>
                <div class="bubble">
                    Xin chào! Tôi là Yuu. Tôi có thể giúp bạn giải toán đơn giản bằng hết khả năng của tôi.
                </div>
            </div>`;
        historyList.innerHTML = '<p class="section-title">Gần đây</p>';
    }

    // ── Auto-resize textarea ──────────────────────────────────────────────────
    userInput.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = this.scrollHeight + "px";
    });

    // ── Enter gửi, Shift+Enter xuống dòng ────────────────────────────────────
    userInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // ── Xử lý sự kiện "Cuộc hội thoại mới" ──────────────────────────────────────
    newChatBtn.addEventListener("click", async function () {
        if (isLoading) return;

        try {
            // Gọi API xóa session cũ phía server
            await fetch(`/history/${SESSION_ID}`, { method: "DELETE" });
        } catch (err) {
            console.error("Không thể xóa session cũ trên server:", err);
        }

        // Cấp phát Session ID hoàn toàn mới cho cuộc hội thoại tiếp theo
        SESSION_ID = "user_" + Math.random().toString(36).slice(2, 8);
        resetChatWorkspace();
        userInput.value = "";
        userInput.style.height = "auto";
    });

    // ── Submit Câu Hỏi ────────────────────────────────────────────────────────
    chatForm.addEventListener("submit", async function (e) {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text || isLoading) return;

        // 1. Hiển thị tin nhắn người dùng lên giao diện
        appendMessage(text, "user");
        userInput.value = "";
        userInput.style.height = "auto";

        // 2. Chỉ tạo mục lịch sử ở sidebar khi người dùng thực sự gửi tin nhắn
        addToHistory(text);

        // 3. Kích hoạt hiệu ứng ba chấm (...) chờ đợi
        const typingId = appendTyping();
        setLoading(true);

        try {
            const res = await fetch("/chat", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ message: text, session_id: SESSION_ID }),
            });

            removeTyping(typingId);

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Lỗi máy chủ khi xử lý dữ liệu.");
            }

            // 4. Nhận dữ liệu từ FastAPI và hiển thị câu trả lời từ MathRAG
            const data = await res.json();
            appendMessage(data.answer, "ai", data.hits, data.sources);

        } catch (err) {
            removeTyping(typingId);
            appendMessage(`⚠️ Lỗi: ${err.message}`, "ai");
        } finally {
            setLoading(false);
        }
    });

    // ── Render Tin Nhắn & Tích hợp KaTeX ──────────────────────────────────────
    function appendMessage(text, sender, hits = 0, sources = []) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender);

        const avatarDiv = document.createElement("div");
        avatarDiv.classList.add("avatar");
        avatarDiv.innerHTML = sender === "user"
            ? '<i class="fa-solid fa-user"></i>'
            : '<i class="fa-solid fa-robot"></i>';

        const bubbleDiv = document.createElement("div");
        bubbleDiv.classList.add("bubble");
        bubbleDiv.innerHTML = text.replace(/\n/g, "<br>");

        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(bubbleDiv);

        // Hiển thị panel trích dẫn nguồn RAG từ `sources` (nếu có)
        if (sender === "ai" && hits > 0 && sources && sources.length > 0) {
            const metaDiv = document.createElement("div");
            metaDiv.style.cssText = "margin-top:8px;display:flex;gap:10px;align-items:center;";

            const hitsSpan = document.createElement("span");
            hitsSpan.style.cssText = "font-size:11px;color:var(--text-muted);";
            hitsSpan.textContent = `${hits} đoạn ngữ cảnh được tìm thấy`;

            const srcBtn = document.createElement("button");
            srcBtn.style.cssText = "font-size:11px;color:var(--accent-color);background:none;border:none;cursor:pointer;text-decoration:underline;";
            srcBtn.textContent = "Xem nguồn";
            srcBtn.type = "button";
            srcBtn.addEventListener("click", () => toggleSources(srcBtn, sources));

            metaDiv.appendChild(hitsSpan);
            metaDiv.appendChild(srcBtn);
            messageDiv.appendChild(metaDiv);
        }

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Tự động phân tích và hiển thị công thức Toán bằng KaTeX
        if (typeof renderMathInElement === "function") {
            renderMathInElement(bubbleDiv, {
                delimiters: [
                    { left: "$$", right: "$$", display: true  },
                    { left: "$",  right: "$",  display: false },
                    { left: "\\[", right: "\\]", display: true  },
                    { left: "\\(", right: "\\)", display: false },
                ],
                throwOnError: false,
            });
        }

        return messageDiv;
    }

    // ── Đóng / Mở Panel Nguồn Trích Dẫn ───────────────────────────────────────
    function toggleSources(btn, sources) {
        const existing = btn.parentElement.nextElementSibling;
        if (existing && existing.classList.contains("sources-panel-inline")) {
            existing.remove();
            btn.textContent = "Xem nguồn";
            return;
        }
        btn.textContent = "Ẩn nguồn";

        const panel = document.createElement("div");
        panel.classList.add("sources-panel-inline");
        panel.style.cssText = "margin-top:8px;background:#1a1a1c;border:1px solid var(--border-color);border-radius:10px;padding:10px 14px;font-size:12px;color:var(--text-muted);max-height:180px;overflow-y:auto;width:100%;box-sizing:border-box;";

        sources.forEach((s, i) => {
            const item = document.createElement("div");
            item.style.cssText = "padding:4px 0;border-bottom:1px solid #2a2a2d;font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
            item.textContent = `[${i + 1}] ${s}`;
            item.title = s;
            panel.appendChild(item);
        });

        btn.parentElement.insertAdjacentElement("afterend", panel);
    }

    // ── Tạo Hiệu Ứng Ba Chấm (...) Khi Chờ Phản Hồi ──────────────────────────────
    let typingCounter = 0;

    function appendTyping() {
        const id = "typing-" + (++typingCounter);

        if (!document.getElementById("blink-style")) {
            const style = document.createElement("style");
            style.id = "blink-style";
            style.textContent = `@keyframes blink{0%,60%,100%{opacity:.3;transform:translateY(0)}30%{opacity:1;transform:translateY(-4px)}}`;
            document.head.appendChild(style);
        }

        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", "ai");
        messageDiv.id = id;

        const avatarDiv = document.createElement("div");
        avatarDiv.classList.add("avatar");
        avatarDiv.innerHTML = '<i class="fa-solid fa-robot"></i>';

        const bubbleDiv = document.createElement("div");
        bubbleDiv.classList.add("bubble");
        bubbleDiv.innerHTML = `
            <span style="display:inline-flex;gap:5px;align-items:center;padding-top:4px;">
                <span style="width:7px;height:7px;border-radius:50%;background:var(--accent-color);animation:blink 1.2s infinite 0s;display:inline-block;"></span>
                <span style="width:7px;height:7px;border-radius:50%;background:var(--accent-color);animation:blink 1.2s infinite 0.2s;display:inline-block;"></span>
                <span style="width:7px;height:7px;border-radius:50%;background:var(--accent-color);animation:blink 1.2s infinite 0.4s;display:inline-block;"></span>
            </span>`;

        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return id;
    }

    function removeTyping(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    // ── Đưa Câu Hỏi Thực Tế Vào Lịch Sử Ở Sidebar ──────────────────────────────
    function addToHistory(text) {
        document.querySelectorAll(".history-item").forEach(i => i.classList.remove("active"));

        const item = document.createElement("div");
        item.classList.add("history-item", "active");
        item.innerHTML = `<i class="fa-regular fa-message"></i> ${text.length > 28 ? text.slice(0, 28) + "…" : text}`;
        item.title = text;

        const sectionTitle = historyList.querySelector(".section-title");
        if (sectionTitle) {
            sectionTitle.insertAdjacentElement("afterend", item);
        } else {
            historyList.appendChild(item);
        }
    }

    // ── Quản Lý State Khi Đang Tải (Loading) ───────────────────────────────────
    function setLoading(val) {
        isLoading = val;
        userInput.disabled = val;
        const sendBtn = document.getElementById("send-btn");
        sendBtn.disabled = val;

        if (val) {
            sendBtn.style.opacity = "0.5";
            sendBtn.style.cursor = "not-allowed";
        } else {
            sendBtn.style.opacity = "1";
            sendBtn.style.cursor = "pointer";
            userInput.focus();
        }
    }
});