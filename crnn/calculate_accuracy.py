import csv

with open('crnn_prediction_results_with_accuracy.csv') as f:
    reader = csv.DictReader(f)
    correct = 0
    total = 0
    char_acc = 0
    similarity = 0
    
    for row in reader:
        correct += 1 if row['is_correct'] == 'True' else 0
        char_acc += float(row['char_accuracy'])
        similarity += float(row['similarity'])
        total += 1

    print(f'전체 정확도: {correct/total:.2%}')
    print(f'문자 단위 정확도: {char_acc/total:.2%}')
    print(f'유사도: {similarity/total:.2%}')
    print(f'총 테스트 샘플 수: {total}') 