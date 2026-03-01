## Arquivos de Dados

Os arquivos intermediários (**data/interim**) são gerados em formato **Parquet** para preservar tipos, reduzir tamanho e acelerar IO.  
Os dados originais permanecem em **data/raw** no formato fornecido **.csv**.

## Exclusão de colunas dos dados originais e criação de features

Para garantir **reprodutibilidade** e **ausência de vazamento temporal**, e minimizar **sobreajuste**, foram excluídas variáveis originais.  
Os dados pessoais foram **anonimizados na origem**, não exigindo tratamento adicional de **PII (Personally Identifiable Information)**.

### Colunas excluídas (critérios)

- **Redundância**
  - A coluna **Nome** foi excluída pois é redundante com **RA (Registro Acadêmico)**, que será mantido apenas como **chave de junção** entre anos e para auditoria.
  - A coluna **INDE (Índice do Desenvolvimento Educacional)** foi excluída pois é derivada de **7 indicadores** mantidos como features e utilizada na determinação da coluna **Pedra**.

- **Texto livre**
  - Como não será utilizada **NLP (Natural Language Processing)**, foram excluídas do baseline as colunas com texto livre:
    - **Destaque IEG**, **Destaque IDA**, **Destaque IPV**.

- **Alta cardinalidade / viés**
  - As colunas **Avaliador1** a **Avaliador6** foram removidas por alta cardinalidade e potencial de viés.

- **Rankings dependentes do conjunto**
  - **Cg**, **Cf**, **Ct** (classificações/rankings) não são usados no baseline porque podem depender da distribuição do grupo no ano e dificultam replicação em inferência individual via API.

## Colunas mantidas como features (ano `t`)

- Variáveis demográficas e de contexto:
  - **Gênero**, **Ano nasc/idade**, **Ano ingresso**, **Instituição de ensino**, **Turma**, **Fase**.
- Indicadores quantitativos do acompanhamento no ano `t`:
  - **IAA (Autoavaliação)**, **IAN (Adequação de Nível)**, **IDA (Desempenho Acadêmico)**, **IEG (Engajamento)**, **IPP (Psicopedagógico)**, **IPS (Psicossocial)**, **IPV (Ponto de Virada)**.
- Notas:
  - **Matemática**, **Português**, **Inglês** (após conversão para numérico).
- Variáveis históricas e recomendações:
  - **Pedra**, **Rec Av1** a **Rec Av4**, **Rec Psicologia** (com tratamento de valores ausentes).

## Criação de features

Além das colunas originais, criamos atributos derivados para capturar padrões temporais e melhorar interpretabilidade:

- **Idade no ano `t`**: calculada a partir de **Ano nasc** (evita dependência de uma coluna específica como “Idade 22”).
- **Tempo de ingresso**: `ano_t - Ano ingresso`.
- **Gap de fase**: `Fase - Fase ideal` (medida direta de defasagem escolar).
- **Codificação ordinal da coluna Pedra**: mapeamento com ordem (ex.: Quartzo < Ágata < Ametista < Topázio), preservando noção de progresso.
- **Cobertura de avaliações**: contagem de recomendações presentes (**Rec Av1..Rec Av4**) e uso de **Nº Av**.

## Regra de não vazamento temporal

A previsão é sempre `t → t+1`.  
Portanto, somente variáveis do ano `t` entram no modelo. Qualquer coluna do ano `t+1` é usada exclusivamente para gerar o alvo (**y**) e para avaliação, nunca como entrada do modelo.
