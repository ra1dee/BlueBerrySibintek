import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import docx
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, FuzzyTermPlugin
import shutil
import string
from nltk.stem import WordNetLemmatizer
import pickle

keras = tf.keras
layers = tf.keras.layers

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer_en = WordNetLemmatizer()


def lemmatize_word(word):
    return lemmatizer_en.lemmatize(word) if word.isalpha() else word


def preprocess_query(query):
    stop_words = set(stopwords.words('russian') + stopwords.words('english'))
    tokens = word_tokenize(query.lower())
    tokens = [lemmatize_word(token) for token in tokens if token not in stop_words and token not in string.punctuation]
    return ' '.join(token for token in tokens if len(token) > 2)


def load_data():
    try:
        data = pd.read_csv('dataset_.csv', encoding='windows-1251', sep=';', on_bad_lines='skip')
        data.columns = data.columns.str.strip()
        print("Названия столбцов:", data.columns)
        print(data.head())

        class_counts = data['label'].value_counts()
        print(f"Распределение классов:\n{class_counts}")
        print(f"Количество классов с одним образцом: {sum(class_counts == 1)}")

        return data, data['Topic'].values, data['label'].values
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None, None, None


def create_model(input_dim, output_dim):
    model = keras.Sequential([
        layers.Embedding(input_dim=input_dim + 1, output_dim=128),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class InstructionMemory:
    def __init__(self):
        self.instructions = {}
        self.documents_content = []
        self.index_dir = "indexdir"
        self.setup_index()

    def setup_index(self):
        if os.path.exists(self.index_dir):
            shutil.rmtree(self.index_dir)
        os.mkdir(self.index_dir)
        schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT)
        self.ix = create_in(self.index_dir, schema)

    def load_docx_files(self, directory):
        writer = self.ix.writer()
        for filename in os.listdir(directory):
            if filename.endswith('.docx'):
                file_path = os.path.join(directory, filename)
                doc = docx.Document(file_path)
                content = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
                full_content = ' '.join(content)
                self.documents_content.append(full_content)
                writer.add_document(title=filename, path=file_path, content=full_content)
        writer.commit()

    def find_instruction_file(self, topic):
        with self.ix.searcher() as searcher:
            qp = QueryParser("content", self.ix.schema)
            qp.add_plugin(FuzzyTermPlugin())
            query = qp.parse(preprocess_query(topic))

            results = searcher.search(query, limit=5)  # Ограничиваем число результатов

            # Если результатов нет, возвращаем файл с минимальной схожестью
            if len(results) == 0:
                print("Прямое совпадение не найдено. Проводим поиск по схожести.")
                query = QueryParser("content", self.ix.schema).parse("*")  # Запрос для поиска всего
                all_results = searcher.search(query)
                if all_results:
                    top_result = max(all_results, key=lambda r: r.score)  # Находим лучший по схожести
                    return [(top_result['title'], top_result['path'], top_result.score)]
                else:
                    return [("Инструкция не найдена", "", 0)]

            # Если результаты найдены, возвращаем их
            return [(result['title'], result['path'], result.score) for result in results]

    def search_instructions(self, query):
        with self.ix.searcher() as searcher:
            query = QueryParser("content", self.ix.schema).parse(query)
            results = searcher.search(query, limit=5)
            return [(result['title'], result['path'], result.score) for result in results]


def find_similar_topics_and_solutions(user_input, data, instruction_memory):
    # Все текстовые данные (топики из docx и из датасета)
    all_texts = instruction_memory.documents_content + list(data['Topic'])
    tokenized_texts = [word_tokenize(preprocess_query(text)) for text in all_texts]
    model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

    def get_text_vector(text):
        words = word_tokenize(preprocess_query(text))
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

    text_vectors = [get_text_vector(text) for text in all_texts]
    user_vector = get_text_vector(user_input)
    similarities = cosine_similarity([user_vector], text_vectors)[0]

    threshold_similarity = 0.1
    unique_topics = {}
    for i in range(len(similarities)):
        if i >= len(instruction_memory.documents_content) and similarities[i] > threshold_similarity:
            topic = all_texts[i]
            if topic not in unique_topics:  # Проверяем на уникальность
                unique_topics[topic] = (data.index[data['Topic'] == topic].tolist()[0],
                                        topic,
                                        str(data.loc[data['Topic'] == topic, 'Solution'].values[0]),
                                        similarities[i])

    similar_topics_with_solutions = list(unique_topics.values())

    instruction_results = instruction_memory.search_instructions(user_input)
    combined_results = similar_topics_with_solutions + [
        (None, f"Инструкция: {title}", f"Файл: {path}", score)
        for title, path, score in instruction_results
    ]

    return sorted(combined_results, key=lambda x: x[3], reverse=True)[:5]



def main():
    data, X, y = load_data()
    if data is None:
        return

    min_samples_per_class = 2
    class_counts = data['label'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    data_filtered = data[data['label'].isin(valid_classes)]

    X = data_filtered['Topic'].values
    y = data_filtered['label'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = keras.preprocessing.sequence.pad_sequences(X_seq)

    X_train, X_test, y_train, y_test = train_test_split(X_pad, y_encoded, test_size=0.2, random_state=42)

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('label_encoder.pickle', 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

    load_existing_model = input("Хотите загрузить существующую модель? (y/n): ").strip().lower()

    if load_existing_model == 'y':
        try:
            model = keras.models.load_model('my_model.keras')
            print("Модель загружена успешно.")
            if input("Хотите дообучить модель? (y/n): ").strip().lower() == 'y':
                model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2,
                          # callbacks=[
                          #     keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                          #     keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
                          # ]
                          )
                model.save('my_model.keras')

        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            model = create_model(len(tokenizer.word_index), len(np.unique(y_encoded)))
            model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2,
                      callbacks=[
                          keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                          keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
                      ])
            model.save('my_model.keras')
    else:
        model = create_model(len(tokenizer.word_index), len(np.unique(y_encoded)))
        model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2,
                  callbacks=[
                      keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                      keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
                  ])
        model.save('my_model.keras')

    instruction_memory = InstructionMemory()
    instruction_memory.load_docx_files(os.getcwd())

    while True:
        topic_to_predict = input("Введите топик (или 'q' для завершения): ")
        if topic_to_predict.lower() == 'q':
            break

        seq = tokenizer.texts_to_sequences([topic_to_predict])
        pad = keras.preprocessing.sequence.pad_sequences(seq, maxlen=X_pad.shape[1])
        predicted_label = label_encoder.inverse_transform([np.argmax(model.predict(pad))])

        print(f"Предсказанная метка: {predicted_label[0]}")

        solution = data.loc[data['Topic'] == topic_to_predict, 'Solution'].values
        print(f"Решение: {solution[0] if len(solution) > 0 else 'Решение не найдено.'}")

        instruction_file = instruction_memory.find_instruction_file(topic_to_predict)
        print(f"Инструкция находится в файле: {instruction_file[0][1] if instruction_file[0][1] else 'Не найдено'}")

        similar_topics_and_solutions = find_similar_topics_and_solutions(topic_to_predict, data, instruction_memory)
        print("Похожие топики и их решения:")
        for index, topic, soln, similarity in similar_topics_and_solutions:
            if index is not None:
                print(f"- Номер строки: {index + 1}, Топик: {str(topic)}, Решение: {str(soln)}, Схожесть: {similarity:.2f}")
            else:
                print(f"- {topic}, {soln}, Схожесть: {similarity:.2f}")


if __name__ == "__main__":
    main()