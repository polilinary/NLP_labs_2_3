# NLP_labs_2_3

## Шаги

Сначала надо скачать датасет в текущию директорию:
```bash
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xf data.tar
```

Установить зависимости
```bash
pip install -r requirements.txt
```

### Модели
Чтобы получить квантованную модель:
```bash
python compress.py
```

Чтобы натренировать модель:

```bash
python train.py
```
Скачать уже готовые модели: 
https://huggingface.co/POLILILILILILI/quantized_model
https://huggingface.co/POLILILILILILI/lora

### Скрипты

Чтобы посчитать размеры:
```bash
python calculate_compression.py 
```
Чтобы отдельно запустить оценку  mmlu:
```bash
python evaluate.py model_name <model> --limit_prompts 0.1
```

Чтобы запустить все для подсчета скоров:
```bash
python lab_evaluation.py
```


