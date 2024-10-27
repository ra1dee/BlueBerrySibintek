document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("submitButton").addEventListener("click", async () => {
        const userInput = document.getElementById("userInput").value;
        const actions = Array.from(document.querySelectorAll('input[name="action"]:checked')).map(checkbox => checkbox.value);



        const resultsContainer = document.getElementById("results");

        try {
            const data = await sendRequest(userInput, actions); // Отправить запрос и дождаться ответа

          /*  // Очистить сообщение "Загрузка..." после получения данных
            resultsContainer.innerHTML = "";

            // Обработка ответа на основе типа запроса
            actions.forEach(action => {
                const resultItem = document.createElement("div");
                resultItem.classList.add("result-item");

                if (action === "service" && data.service) {
                    resultItem.innerHTML = `<strong>Сервис:</strong> ${data.service.name}`;
                    if (data.service.alternatives) {
                        resultItem.innerHTML += `<br><strong>Альтернативные сервисы:</strong><ul>`;
                        data.service.alternatives.forEach(alt => {
                            resultItem.innerHTML += `<li>${alt.name} (Вероятность: ${alt.probability}%)</li>`;
                        });
                        resultItem.innerHTML += `</ul>`;
                    }
                } else if (action === "similarRequests" && data.similarRequests) {
                    resultItem.innerHTML = `<strong>Похожие обращения:</strong><ul>`;
                    data.similarRequests.forEach(request => {
                        resultItem.innerHTML += `<li>Номер обращения: ${request.id}, Текст решения: ${request.solution} (Метрика подобия: ${request.similarityScore}%)</li>`;
                    });
                    resultItem.innerHTML += `</ul>`;
                } else if (action === "instructions" && data.instructions) {
                    resultItem.innerHTML = `<strong>Инструкции:</strong><ul>`;
                    data.instructions.forEach(instruction => {
                        resultItem.innerHTML += `<li>Файл: ${instruction.filename}, Выдержка: ${instruction.excerpt}</li>`;
                    });
                    resultItem.innerHTML += `</ul>`;
                }

                resultsContainer.appendChild(resultItem);
            });*/
        } catch (error) {
            resultsContainer.innerHTML = "<p>Ошибка при получении данных.</p>";
            console.error("Ошибка:", error);
        }
    });
});

async function sendRequest(topic, actions) {
    const url = "http://127.0.0.1:8080/topic/" + encodeURIComponent(topic);
    const Http = new XMLHttpRequest();
    Http.open("GET", url);
    Http.send();
    console.log("Отправка GET-запроса на URL:", url);

}
function refresh() {
    location.reload()
}
