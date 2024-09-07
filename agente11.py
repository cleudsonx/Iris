import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, fbeta_score

def run_agente1(y_test, y_pred):
    # Gerar o gráfico F1 curve
    plt.figure()
    plt.plot(y_test, y_pred, label='F1 curve')
    plt.ylabel('Score')
    plt.title('F1 curve')
    plt.legend()
    plt.savefig('static/f1_curve.png')
    plt.close()

    # Calcular e salvar as métricas
    f1 = f1_score(y_test, y_pred, average='macro')
    fbeta_05 = fbeta_score(y_test, y_pred, beta=0.5, average='macro')
    fbeta_2 = fbeta_score(y_test, y_pred, beta=2, average='macro')

    with open('static/metrics.txt', 'w') as f:
        f.write(f"F1 Score: {f1}\n")
        f.write(f"F-beta Score (beta=0.5): {fbeta_05}\n")
        f.write(f"F-beta Score (beta=2): {fbeta_2}\n")

    return f1, fbeta_05, fbeta_2