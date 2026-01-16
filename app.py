import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import os
from io import StringIO
import json

# Configuração da página
st.set_page_config(
    page_title="Assistente de Análise de Dados IA",
    page_icon="🤖",
    layout="wide"
)

# Título e descrição
st.title("🤖 Assistente de Análise de Dados Inteligente")
st.markdown("""
Faça upload de um arquivo CSV ou Excel e converse com seus dados em linguagem natural!
O assistente IA irá analisar, visualizar e extrair insights automaticamente.
""")

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Input da API Key do Groq
    api_key = st.text_input(
        "🔑 Chave API Groq",
        type="password",
        help="Obtenha sua API key em: https://console.groq.com"
    )
    
    st.divider()
    
    # Seleção do modelo ATUALIZADA
    model = st.selectbox(
        "🧠 Modelo de IA",
        [
            "llama-3.1-8b-instant",         # Mais rápido
            "llama-3.2-90b-text-preview",   # Mais poderoso (beta)
            "llama-3.2-1b-preview",         # Leve e rápido
            "gemma2-9b-it"                  # Alternativa
        ],
        index=0,
        help="Modelos ativos do Groq - mixtral-8x7b-32768 foi descontinuado"
    )
    
    # Temperatura para criatividade
    temperature = st.slider(
        "🎭 Temperatura (criatividade)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Valores mais altos = mais criativo, mais baixo = mais focado"
    )
    
    st.divider()
    
    # Exemplo de perguntas
    st.subheader("💡 Exemplos de perguntas:")
    st.markdown("""
    - "Quais são as principais estatísticas descritivas?"
    - "Existe correlação entre [coluna1] e [coluna2]?"
    - "Mostre a distribuição de [coluna]"
    - "Quais são os outliers nos dados?"
    - "Crie um gráfico de linha para [coluna] ao longo do tempo"
    - "Agrupe os dados por [coluna] e calcule médias"
    """)
    
    st.divider()
    st.caption("Powered by Groq & Streamlit")

# Inicializar cliente Groq
@st.cache_resource
def init_groq_client(api_key):
    if api_key and api_key.strip():
        try:
            return Groq(api_key=api_key.strip())
        except Exception as e:
            st.error(f"Erro ao inicializar cliente Groq: {str(e)}")
            return None
    return None

client = init_groq_client(api_key)

# Função para análise básica do dataset
def analyze_dataframe(df):
    """Realiza análise básica do dataframe"""
    analysis = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "sample_data": df.head(5).to_dict('records')
    }
    return analysis

# Função para gerar visualizações automáticas
def generate_auto_visualizations(df, analysis):
    """Gera visualizações automáticas baseadas nos dados"""
    viz_suggestions = []
    
    # Para colunas numéricas
    numeric_cols = analysis['numeric_columns']
    
    if len(numeric_cols) >= 1:
        # Histograma para a primeira coluna numérica
        try:
            fig = px.histogram(df, x=numeric_cols[0], 
                              title=f"Distribuição de {numeric_cols[0]}",
                              nbins=30)
            viz_suggestions.append(("Histograma", fig))
        except Exception as e:
            st.warning(f"Não foi possível criar histograma: {str(e)}")
    
    if len(numeric_cols) >= 2:
        # Scatter plot entre duas colunas numéricas
        try:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                           title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
            viz_suggestions.append(("Scatter Plot", fig))
        except:
            pass
    
    # Para colunas categóricas
    cat_cols = analysis['categorical_columns']
    if cat_cols and numeric_cols:
        # Bar chart de média por categoria
        try:
            # Escolher coluna categórica com menos valores únicos para melhor visualização
            cat_col = min(cat_cols, key=lambda x: df[x].nunique())
            fig = px.bar(df.groupby(cat_col)[numeric_cols[0]].mean().reset_index(),
                        x=cat_col, y=numeric_cols[0],
                        title=f"Média de {numeric_cols[0]} por {cat_col}")
            viz_suggestions.append(("Bar Chart", fig))
        except:
            pass
    
    return viz_suggestions

# Função para chamar a API Groq
def query_groq(client, model, prompt, data_context, temperature=0.7):
    """Envia consulta para a API Groq"""
    
    system_prompt = f"""Você é um assistente especializado em análise de dados.
    
    CONTEXTO DOS DADOS:
    {data_context}
    
    INSTRUÇÕES:
    1. Analise a pergunta do usuário sobre os dados
    2. Forneça insights baseados nos dados fornecidos
    3. Sugira visualizações relevantes
    4. Seja conciso e direto
    5. Se a pergunta envolver cálculos, explique como eles seriam feitos
    6. Se faltarem informações nos dados, explique isso claramente
    
    Responda em português brasileiro.
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        # Tratamento específico para modelo descontinuado
        if "model_decommissioned" in error_msg or "mixtral-8x7b-32768" in error_msg:
            return "❌ **Erro: O modelo selecionado foi descontinuado.**\n\n🔧 **Solução:** Selecione outro modelo na sidebar, como:\n- `llama-3.1-8b-instant` (mais rápido)\n- `gemma2-9b-it` (alternativa)"
        elif "authentication" in error_msg.lower():
            return "❌ **Erro de autenticação.** Verifique se sua API Key do Groq está correta e ativa."
        elif "rate limit" in error_msg.lower():
            return "⚠️ **Limite de requisições atingido.** A conta gratuita do Groq tem limites. Tente novamente em alguns minutos."
        else:
            return f"❌ **Erro na consulta à API:** {error_msg}\n\n💡 **Sugestões:**\n1. Verifique sua conexão com a internet\n2. Tente um modelo diferente\n3. Verifique se a API Key está correta"

# Área principal da aplicação
tab1, tab2, tab3 = st.tabs(["📤 Upload de Dados", "📊 Análise Automática", "💬 Chat com Dados"])

# Tab 1: Upload de dados
with tab1:
    st.header("1. Faça upload dos seus dados")
    
    uploaded_file = st.file_uploader(
        "Escolha um arquivo CSV ou Excel",
        type=['csv', 'xlsx', 'xls'],
        help="Tamanho máximo: 200MB"
    )
    
    if uploaded_file is not None:
        try:
            # Ler o arquivo baseado na extensão
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Limpeza básica: remover colunas totalmente vazias
            df = df.dropna(axis=1, how='all')
            
            # Salvar dataframe na session state
            st.session_state['df'] = df
            st.session_state['file_name'] = uploaded_file.name
            
            st.success(f"✅ Arquivo '{uploaded_file.name}' carregado com sucesso!")
            
            # Mostrar preview
            with st.expander("📋 Visualizar dados (primeiras 10 linhas)"):
                st.dataframe(df.head(10), use_container_width=True)
                
            # Mostrar informações básicas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Linhas", df.shape[0])
            with col2:
                st.metric("Colunas", df.shape[1])
            with col3:
                st.metric("Valores ausentes", df.isnull().sum().sum())
            with col4:
                st.metric("Tamanho", f"{uploaded_file.size / 1024:.1f} KB")
                
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {str(e)}")
            st.info("💡 Dica: Verifique se o arquivo está no formato correto.")

# Tab 2: Análise automática
with tab2:
    st.header("2. Análise Automática dos Dados")
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Realizar análise
        analysis = analyze_dataframe(df)
        
        # Layout em colunas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Informações das Colunas")
            columns_df = pd.DataFrame({
                'Coluna': df.columns,
                'Tipo': df.dtypes.values,
                'Valores Únicos': [df[col].nunique() for col in df.columns],
                'Valores Ausentes': df.isnull().sum().values
            })
            st.dataframe(columns_df, use_container_width=True)
        
        with col2:
            st.subheader("📈 Estatísticas Descritivas")
            if analysis['numeric_columns']:
                st.dataframe(df[analysis['numeric_columns']].describe(), 
                           use_container_width=True)
            else:
                st.info("Nenhuma coluna numérica encontrada para análise estatística.")
        
        # Visualizações automáticas
        st.subheader("🎨 Visualizações Sugeridas")
        viz_suggestions = generate_auto_visualizations(df, analysis)
        
        if viz_suggestions:
            # Mostrar 2 visualizações por linha
            for i in range(0, len(viz_suggestions), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(viz_suggestions):
                        name, fig = viz_suggestions[i + j]
                        with cols[j]:
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Faça upload de dados com colunas numéricas ou categóricas para visualizações automáticas.")
        
    else:
        st.info("📁 Faça upload de um arquivo na aba 'Upload de Dados' para ver a análise automática.")

# Tab 3: Chat com dados
with tab3:
    st.header("3. Chat com seus Dados")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Por favor, faça upload de um arquivo na primeira aba para começar a conversar.")
        st.stop()
    
    if not api_key or not api_key.strip():
        st.error("🔑 Por favor, insira sua API Key do Groq na sidebar para usar o chat.")
        st.info("💡 Obtenha uma API key gratuita em: https://console.groq.com")
        st.stop()
    
    if client is None:
        st.error("❌ Não foi possível conectar à API do Groq. Verifique sua API Key.")
        st.stop()
    
    df = st.session_state['df']
    
    # Preparar contexto dos dados
    data_context = f"""
    Dataset: {st.session_state.get('file_name', 'Arquivo carregado')}
    Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas
    Colunas: {', '.join(df.columns.tolist())}
    
    Tipos de dados:
    {df.dtypes.to_string()}
    
    Amostra dos dados (5 primeiras linhas):
    {df.head().to_string()}
    
    Estatísticas descritivas:
    {df.describe().to_string() if not df.select_dtypes(include=['number']).empty else 'Sem colunas numéricas'}
    """
    
    # Inicializar histórico de chat
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"👋 Olá! Estou pronto para analisar seu dataset '{st.session_state.get('file_name', '')}'. "
                                           f"📊 **Dimensões:** {df.shape[0]} linhas × {df.shape[1]} colunas\n\n"
                                           f"🔍 **Principais colunas:** {', '.join(df.columns.tolist()[:5])}{'...' if len(df.columns) > 5 else ''}\n\n"
                                           f"💡 **O que você gostaria de saber sobre esses dados?**"}
        ]
    
    # Mostrar histórico de chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input do usuário
    if prompt := st.chat_input("Digite sua pergunta sobre os dados..."):
        # Adicionar mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Mostrar indicador de processamento
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 Analisando seus dados...")
            
            try:
                # Chamar API Groq
                response = query_groq(client, model, prompt, data_context, temperature)
                
                # Exibir resposta
                message_placeholder.markdown(response)
            except Exception as e:
                error_msg = f"❌ **Erro durante a análise:** {str(e)}\n\n💡 **Sugestões:**\n1. Tente um modelo diferente\n2. Verifique sua conexão\n3. Reduza o tamanho do dataset"
                message_placeholder.markdown(error_msg)
                response = error_msg
        
        # Adicionar resposta ao histórico
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Controles na parte inferior
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Limpar Conversa", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Conversa limpa! Como posso ajudar com seus dados agora?"}
            ]
            st.rerun()
    
    with col2:
        if st.button("🔄 Atualizar Modelo", use_container_width=True, 
                    help="Recarregar com as configurações atuais"):
            st.info(f"Usando modelo: {model} com temperatura: {temperature}")
            st.rerun()

# Rodapé
st.divider()

# Seção de ajuda
with st.expander("❓ Precisa de ajuda?"):
    st.markdown("""
    ### 🔧 **Problemas Comuns e Soluções:**
    
    1. **Erro 'model_decommissioned':**
       - O modelo `mixtral-8x7b-32768` foi descontinuado
       - Use o `llama-3.1-8b-instant`
    
    2. **Erro de API Key:**
       - Obtenha chave gratuita em [console.groq.com](https://console.groq.com)
       - Copie toda a chave (começa com `gsk_`)
       - Não inclua espaços extras
    
    3. **Limite de requisições:**
       - Conta gratuita tem limite de requests por minuto
       - Aguarde alguns segundos e tente novamente
    
    4. **Arquivo não carrega:**
       - Verifique formato (CSV, Excel)
       - Tamanho máximo: 200MB
       - Sem caracteres especiais no nome
    """)

st.caption("""
🔧 **Dicas de uso:**
1. Use `llama-3.1-8b-instant` para melhores resultados
2. Comece com perguntas simples como "mostre estatísticas básicas"
3. Ajuste a temperatura: mais baixa para respostas mais precisas
4. Para datasets grandes, use o modelo `llama-3.1-8b-instant` (mais rápido)
""")

st.caption(f"📱 Modelo atual: **{model}** | 🌡️ Temperatura: **{temperature}**")
