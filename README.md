

*Arquivos de Dados*
Os arquivos intermediários (data/interim) são gerados em formato Parquet para preservar tipos, reduzir tamanho e acelerar IO. Os dados originais permanecem em data/raw no formato fornecido.


*Exclusão de colunas dos dados originais e criação de features*

Para garantir reprodutibilidade e ausência de vazamento temporal, e minimizar sobreajuste, foram excluídas variáveis originais. Os dados pessoais foram anonimizados na origem, não exigindo tratamento adicional de PII (Personally Identifiable Information).

Colunas excluídas conforme critérios:

*Redundância:* col Nome foi excluído pois é redundante com col RA (Registro Acadêmico), que será mantido apenas como chave de junção entre anos e para auditoria. A col INDE (Índice do Desenvolvimento Educacional) foi excluída pois é derivada de 7 indicadores mantidos como features e utilizada na determinação do valor da col Pedra.

*Texto livre:* como não será utilizada NLP (Natural Language Processing) foram excluídas do baseline as colunas com texto livre: Destaque IEG, Destaque IDA, Destaque IPV. 

*Alta cardianalidade/viés:* cols Avaliador1 a Avaliador6 foram removidas por alta cardinalidade e potencial de viés.

*Rankings dependentes do conjunto:* Cg, Cf, Ct (classificações/rankings) não são usados no baseline porque podem depender da distribuição do grupo no ano e dificultam replicação em inferência individual via API.

*Colunas mantidas como features (ano t)*

Colunas de variáveis demográficas e de contexto (Gênero, Ano nasc/idade, Ano ingresso, Instituição de ensino, Turma, Fase) e indicadores quantitativos do acompanhamento no ano t ((IAA (Autoavaliação), IAN (Adequação de Nível), IDA (Desempenho Acadêmico),   IEG (Engajamento), IPP (Psicopedagógico), IPS (Psicossocial) IPV (Ponto de Virada)), além de notas como Matemática, Português e Inglês após conversão para numérico). Variáveis históricas como Pedra e recomendações (Rec Av1 a Rec Av4 e Rec Psicologia) também são consideradas, com tratamento de valores ausentes.

*Criação de features*

Além das colunas originais, criamos atributos derivados para capturar padrões temporais e melhorar interpretabilidade:

*Idade no ano t:* calculada a partir de Ano nasc (evita dependência de uma coluna “Idade 22” específica).

*Tempo de ingresso:* ano_t - Ano ingresso.

*Gap de fase:* Fase - Fase ideal (medida direta de defasagem escolar).

*Codificação ordinal da col Pedra:* mapeamos categorias para uma ordem (ex.: Quartzo < Ágata < Ametista < Topázio), preservando a noção de progresso.

*Cobertura de avaliações:* contagem de recomendações presentes (Rec Av1..Rec Av4) e uso de Nº Av.

*Regra de não vazamento temporal*

A previsão é sempre t → t+1. Portanto, somente variáveis do ano t entram no modelo. Qualquer coluna do ano t+1 é usada exclusivamente para gerar o alvo (y) e para avaliação, nunca como entrada do modelo.
