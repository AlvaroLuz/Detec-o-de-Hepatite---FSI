# Projeto 1 - FSI
### Álvaro Veloso Cavalcanti Luz - 180115391

## Informações fundamentais:

Este é meu projeto 1 para a disciplina de Fundamentos de Sistemas Inteligentes. Este projeto foi desenvolvido na linguagem 

> python 3.8

Para seu desenvolvimento foram usadas as seguintes bibliotecas com as correspondentes versões:
> numpy = 1.22.2\
> pandas = 1.4.3\
> matplotlib = 3.5.2

O programa inteiro foi desenvolvido utilizando o site "replit" para execução do programa.

Informações de uso do programa constam no menu exibido durante a execução.

## Método escolhido:

Para o cumprimento de todos os requisitos presentes na especificação do trabalho, o programa foi desenvolvido cumprindo as diretrizes estabelecidas da seguinte forma. 

Primeiramente, os dados são coletados no link fornecido para o github através da url. Após isso, os dados de ambos arquivos são concatenados em um único dataframe. Em seguida, os dados passam por um Imputer, que insere valores para substituirem os "?" presentes nos dados originais. O modelo de Imputer selecionado estima o valor que falta baseado nos K vizinhos mais próximos. Depois que temos a tabela completamente preenchida, só então, gera-se os dados de média e desvio padrão para cada parâmetro. Estes dados são exibidos no terminal.

Então, os dados são separados e começa-se o processo de validação cruzada. Durante essa etapa, são gerados gráficos AUC-ROC para cada fold gerada pelo nosso _cross-validator_, no caso o _StratifiedKFold_. Além disso, durante a mesma etapa, a predição feita pelo classificador é salva em um vetor. Assim, depois que acabam as 10 rodadas de validação cruzada, gera-se um gráfico médio AUC-ROC para todas as rodadas e gera-se uma matriz de confusão geral para todas as 10 rodadas. Ambos são exibidos em figuras neste programa.

Por fim, exibe-se no terminal os dois parâmetros mais relevantes para o modelo de classificação selecionado no programa. Junto do valor médio de relevância para diminuir a entropia no conjunto.
