document.addEventListener("DOMContentLoaded", function () {
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const chatMessages = document.getElementById("chat-messages");

    // Tự động giãn chiều cao của ô nhập liệu khi gõ nhiều dòng
    userInput.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight) + "px";
    });

    // Xử lý khi nhấn nút gửi tin nhắn
    chatForm.addEventListener("submit", function (e) {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        // 1. Hiển thị tin nhắn người dùng
        appendMessage(text, "user");
        userInput.value = "";
        userInput.style.height = "auto"; // Reset chiều cao ô nhập

        // 2. Giả lập hiệu ứng AI đang "suy nghĩ" rồi trả lời bài toán
        setTimeout(() => {
            let aiResponse = "Tôi đã nhận được bài toán của bạn. Đây là lời giải mẫu chi tiết:";
            
            // Nếu người dùng hỏi về tích phân, phương trình hoặc đạo hàm, hiển thị công thức LaTeX mẫu toán học
            if (text.toLowerCase().includes("tích phân") || text.toLowerCase().includes("int")) {
                aiResponse += `<br><br>Áp dụng công thức tích phân từng phần:<br> 
                $$\\int x \\cdot e^x \\, dx = x \\cdot e^x - \\int e^x \\, dx = e^x(x - 1) + C$$`;
            } else {
                aiResponse += `<br><br>Giả sử phương trình bậc 2 có dạng: $$ax^2 + bx + c = 0$$ <br> Ta tính biệt thức: $$\\Delta = b^2 - 4ac$$`;
            }
            
            appendMessage(aiResponse, "ai");
        }, 800);
    });

    // Hàm tạo và chèn cấu trúc HTML của tin nhắn vào khung chat
    function appendMessage(text, sender) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender);

        const avatarDiv = document.createElement("div");
        avatarDiv.classList.add("avatar");
        avatarDiv.innerHTML = sender === "user" ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';

        const bubbleDiv = document.createElement("div");
        bubbleDiv.classList.add("bubble");
        bubbleDiv.innerHTML = text;

        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(messageDiv);

        // Cuộn xuống đáy màn hình khi có tin nhắn mới
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Gọi thư viện KaTeX dịch các đoạn chữ nằm trong $$ hoặc $ thành ký tự toán học chuẩn
        if (typeof renderMathInElement === "function") {
            renderMathInElement(bubbleDiv, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false}
                ],
                throwOnError : false
            });
        }
    }
});