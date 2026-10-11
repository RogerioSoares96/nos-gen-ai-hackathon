## A tua tarefa neste passo: verificação de contexto

Ignora a instrução de devolver o documento anonimizado. Um detetor marcou
como sensíveis os dados listados mais abaixo, segundo estas categorias:

$category_block

Dados marcados ([id da categoria] "dado" e a linha onde aparece):
$findings_block

Faz duas verificações, sem nunca reescrever o texto:

1. "additions": palavras ou expressões que, mesmo com os dados acima
   ocultados, continuam a revelar ou permitem deduzir a pessoa ou o dado
   (ex.: uma descrição que só se aplica a um local já ocultado). Usa o
   mesmo formato dos dados marcados: "text" e "context" copiados carácter
   a carácter do documento, "category" e "reason".
2. "rejections": o "text" exato de cada dado marcado que afinal não é
   sensível (ex.: o nome da instituição no título, a data ou a referência
   do relatório).

Se não houver nada a acrescentar ou rejeitar, devolve listas vazias.
