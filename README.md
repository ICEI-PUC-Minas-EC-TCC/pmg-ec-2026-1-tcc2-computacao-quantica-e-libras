# Reconhecimento de Sinais Estáticos da Datilologia de Libras

**Curso:** Engenharia da Computação - PUC Minas

**Autores:**
- Nome: Iander Marques Pereira Costa
- Nome: Isabela Ferreira Scarabelli
- Nome: Milleny Teixeira de Souza
- Nome: Pedro Henrique de Almeida Santos
- Nome: Rafael Felipe Silva Pereira

**Orientadores:**
- Orientador: Felipe Augusto Lara Soares
- Co-orientador: André Tanus Cesário de Souza

---

## Resumo

Segundo o IBGE, mais de 10 milhões de pessoas no Brasil apresentam algum grau de
surdez. A Libras (Língua Brasileira de Sinais) é oficialmente reconhecida como meio de
expressão e comunicação. No entanto, a falta de conhecimento dessa língua pela popula-
ção resulta em barreiras comunicacionais significativas. Diante desse cenário, este trabalho
tem como objetivo investigar o uso de diferentes técnicas de análise de imagens, utilizando
algoritmos de computação clássica e de computação quântica para o desenvolvimento de
sistemas de reconhecimento de sinais estáticos da datilologia (alfabeto manual) de Libras.
Para isso, foi proposto um pipeline estruturado em quatro etapas: Coleta e Seleção de
Conjuntos de Dados, Pré-processamento, Experimentos e Comparação de Resultados. Foi
utilizado um conjunto de dados composto por imagens estáticas da datilologia de Libras
sobre o qual foram avaliadas quatro abordagens de classificação: SVM clássico, rede neu-
ral quântica híbrida, QSVC com kernel quântico e PegasosQSVC. O modelo SVM clássico
apresentou o melhor desempenho geral, com F1 macro de 99,78%. Entre os modelos quân-
ticos e híbridos, o QSVC com kernel quântico obteve o melhor resultado, alcançando F1
macro de 96,08%, seguido pela QNN híbrida, com 94,70%, e pelo PegasosQSVC, com
90,60%. Embora não tenham superado o desempenho do SVM clássico, os modelos quân-
ticos e híbridos apresentaram resultados satisfatórios, indicando sua viabilidade técnica
para a classificação de sinais estáticos da datilologia de Libras.

---

## Objetivos

- Investigar e comparar abordagens de classificação clássicas e quânticas para reconhecimento de sinais estáticos da datilologia de Libras.
- Implementar um pipeline reprodutível envolvendo coleta, pré-processamento, experimentos e análise dos resultados.
- Avaliar desempenho por métricas padronizadas (Acurácia, F1 macro) e estatísticas de validação cruzada.

## Metodologia (Pipeline)

1. Coleta e Seleção de Conjuntos de Dados
   - Utilização de imagens estáticas da datilologia de Libras. Conjunto principal: `Dataset/features_geometry_70.csv`.
2. Pré-processamento
   - Extração e seleção de features geométricas a partir das imagens.
   - Normalização/escala.
3. Experimentos
   - Implementação e treino de quatro abordagens: SVM clássico, QNN híbrida, QSVC com kernel quântico e PegasosQSVC.
   - Validação com Stratified K-Fold e métricas consolidadas por fold.
4. Comparação de Resultados
   - Cálculo de média, desvio padrão (σ) e erro de validação cruzada (CV_error) para Acurácia e F1 macro.

## Modelos Avaliados

- **SVM Clássico**
- **QNN Híbrida**
- **QSVC com kernel Quântico**
- **PegasosQSVC**
  
## Conjunto de Dados e Código

- Os dados de features geométricas processadas estão em `Dataset/features_geometry_70.csv`.
- Notebooks e scripts principais encontram-se em `Modelos/`.

## Resultados (Resumo)

A tabela abaixo resume os resultados obtidos para cada abordagem avaliada.

| Medida / Modelo                         | Clássico | QNN Híbrida | QSVC com kernel Quântico | PegasosQSVC |
|-----------------------------------------|:--------:|:-----------:|:------------------------:|:-----------:|
| Acurácia média                          | 99,78%   | 94,67%      | 96,09%                  | 90,81%      |
| F1 macro                                | 99,78%   | 94,70%      | 96,08%                  | 90,60%      |
| CV_error (Acurácia)                     | 0,22%    | 5,33%       | 3,91%                   | 9,19%       |
| CV_error (F1 macro)                     | 0,22%    | 5,30%       | 3,92%                   | 9,40%       |
| σ Acurácia                              | 0,03%    | 0,74%       | 0,96%                   | 1,63%       |
| σ F1 macro                              | 0,03%    | 0,73%       | 0,99%                   | 0,99%       |

## Conclusões

- O SVM clássico obteve o melhor desempenho geral nas métricas avaliadas neste conjunto de dados.
- Modelos quânticos e híbridos mostraram desempenho competitivo, com o QSVC quântico apresentando o melhor resultado entre eles.
- Resultados indicam viabilidade técnica das abordagens quânticas para o problema, embora seja necessário mais trabalho de engenharia e otimização para superar os métodos clássicos em datasets deste porte.
