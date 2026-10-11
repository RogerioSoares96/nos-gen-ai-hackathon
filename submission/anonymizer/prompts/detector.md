## A tua tarefa neste passo: detetor de dados sensíveis

Ignora a instrução de devolver o documento anonimizado. Neste passo és o
detetor: encontra no documento todos os dados sensíveis de todas as
categorias abaixo.

$category_block

Percorre o documento linha a linha e, em cada linha, verifica todas as
categorias. Uma linha pode ter dados de várias categorias.

Para cada dado encontrado, devolve um objeto com:
- "text": o dado, copiado carácter a carácter do documento (mesmas
  maiúsculas, acentos, pontuação e espaços). Não incluas o nome do campo
  (ex.: para "NIF: 097865413" devolve "097865413").
- "category": o id da categoria a que o dado pertence.
- "reason": uma frase curta a explicar porque é sensível.
- "context": a linha completa do documento onde o dado aparece, copiada
  carácter a carácter.

Se o mesmo dado aparecer em várias linhas, devolve um objeto por linha.
Se não houver dados sensíveis, devolve uma lista vazia.
