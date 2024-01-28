import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from textblob import TextBlob
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')


class Recommendation():
    """
    Esta classe possui todos os métodos que usaremos para analisar os dados, criar os modelos e valida-los.
    """
    
    def __init__(self):
        pass
    
    
    def open_data(self, f_name):
        """
        Função para importar o arquivo CSV
        """ 
              
        data = pd.read_csv(
            os.getcwd().replace('/' or '//', r'\\').replace('code', 'data')
            .replace('modeling', 'raw') + f'\\{f_name}'
        )
        
        return data
    
    def create_data_dict(self):
        """
        Método para criar o dicionário de dados
        """
        
        data_dict = {
            'Nome das variáveis' : [
                'product_id', 'product_name', 'category', 'discounted_price', 'actual_price', 
                'discount_percentage', 'rating', 'rating_count', 'about_product', 'user_id', 
                'user_name', 'review_id', 'review_title', 'review_content', 'img_link', 'product_link'
            ],
            
            'Tipo de dado' : ['object'] * 16,
            
            'Descrição' : [
                'Identificador único para cada produto',
                'Nome do produto',
                'Categoria do produto',
                'Preço com desconto do produto',
                'Preço original do produto antes dos descontos',
                'Porcentagem do desconto concedido no produto',
                'Classificação média dada ao produtopelos usuários',
                'Número de usuários que avaliaram o produto',
                'Descrição ou detalhes sobre o produto',
                'Identificador exclusivo para cado usuário que escreveu a avaliação',
                'Nome do usuário que escreveu a avaliação',
                'Identificador exclusivo para cada avaliação de usuário',
                'Título curto ou resumo de avaliação do usuário',
                'Conteúdo completo da avaliação do usuário',
                'URL link para imagem do produto',
                'URL link para a página do produto no site oficial da Amazon'
            ]
        }
        
        return data_dict
    
    
    def transform_to_real(self,data):
        '''
        Método que transforma os valores de Rupias Indianas para Reais
        '''
        
        # convertendo Rupias para Reais
        data['discounted_price'] = data['discounted_price'] / 17
        data['actual_price'] = data['actual_price'] / 17

        # reduzindo casas decimais
        data['discounted_price'] = data['discounted_price'].map('{:.2f}'.format).astype(float)
        data['actual_price'] = data['actual_price'].map('{:.2f}'.format) .astype(float)

        # transformando porcentagem
        data['discount_percentage'] = data['discount_percentage'] / 100
        
        return data
    
    
    def remove_ec(self, data):
        '''
        Método para remover carcteres especiais de colunas numéricas
        '''
        
        data['rating'] = data['rating'].replace('', '0') 
        data['rating'] = data['rating'].replace('[^\d.]', '', regex=True)

        data['discounted_price'] = data['discounted_price'].replace('', '0') 
        data['discounted_price'] = data['discounted_price'].replace('[^\d.]', '', regex=True)

        data['actual_price'] = data['actual_price'].replace('', '0') 
        data['actual_price'] = data['actual_price'].replace('[^\d.]', '', regex=True)

        data['discount_percentage'] = data['discount_percentage'].replace('[^\d.]', '', regex=True)
        data['rating_count'] = data['rating_count'].replace('[^\d.]', '', regex=True)

        # convertendo para float
        data['rating'] = data['rating'].astype(float)
        data['discounted_price'] = data['discounted_price'].astype(float)
        data['actual_price'] = data['actual_price'].astype(float)
        data['discount_percentage'] = data['discount_percentage'].astype(float)
        data['rating_count'] = data['rating_count'].astype(float)
        
        return data
    
    
    def create_df_insight(self, data):
        '''
        Método para analisar os principais elementos de cada coluna.
        '''
        unique_products_count = data['product_id'].nunique()
        average_price = data['actual_price'].mean()
        best_selling_prod = data.loc[data['rating_count'].idxmax()]
        least_sellings_prod = data.loc[data['rating_count'].idxmin()]
        top_rated_prod = data.loc[data['rating'].idxmax()]
        low_rated_prod = data.loc[data['rating'].idxmin()]
        most_expensive_product = data.loc[data['actual_price'].idxmax()]
        cheapest_product = data.loc[data['actual_price'].idxmin()]
        highest_discount_product = data.loc[data['discount_percentage'].idxmax()]
        avg_rating_count = data.groupby('product_id')['rating_count'].mean().mean()

        data_analysis = pd.DataFrame({
            'Questão':[
                'Número de produtos unicos',
                'Preço médio',
                'Produto mais vendido',
                'Produto menos vendido',
                'Produto mais avaliado',
                'Produto menos avaliado',
                'Produto mais caro',
                'Produto mais barato',
                'Produto com maior desconto',
                'Média de classificalção para cada produto'
            ], 
            
            'Resposta' :[
                unique_products_count,
                round(average_price,2),
                best_selling_prod['product_name'],
                least_sellings_prod['product_name'],
                top_rated_prod['product_name'],
                low_rated_prod['product_name'],
                most_expensive_product['product_name'],
                cheapest_product['product_name'],
                highest_discount_product['product_name'],
                avg_rating_count
            ],
            
            'Preço Atual' :[
                None,
                None, 
                best_selling_prod['actual_price'],
                least_sellings_prod['actual_price'],
                top_rated_prod['actual_price'],
                low_rated_prod['actual_price'],
                most_expensive_product['actual_price'],
                cheapest_product['actual_price'],
                highest_discount_product['actual_price'],
                None
            ]
        })
        
        return data_analysis
    
    
    def top_product(self, data):
        '''
        Método para mostrar os 10 produtos mais vendidos. 
        '''
        top_prod = data.sort_values(by='rating_count', ascending=False).head(10)

        # selecionando apenas colunas relevantes
        top_prod = top_prod[['product_name', 'rating', 'rating_count']]
        top_prod.reset_index(drop=True, inplace=True)
        
        return top_prod
    
    def top_categories(self, data):
        '''
        Método que retorna as 10 principais categorias, e sua contagem.
        '''
        categories = data['category'].str.split('|').explode()
        category_counts = Counter(categories)
        category_df = pd.DataFrame(category_counts.items(), columns = ['Category', 'Count']).sort_values(by='Count', ascending=False)

        top_categories = category_df.head(10)
        return top_categories
    
    
    def avaliation_graphic(self, data):
        '''
        Método que calcula a proporção de avaliações x preço, 
        e avaliações x preço com desconto, amostrado em um 
        gráfico de scatterplot
        '''
        
        fig, axes = plt.subplots(1,2, figsize=(18,6))

        # scatterplot do preço atual X avaliações
        sns.scatterplot(ax=axes[0], data=data, x='actual_price', y='rating', size='rating_count', alpha=0.5)
        axes[0].set_title('Preço Atual X Avaliações')
        axes[0].set_xlabel('Preço Atual')
        axes[0].set_ylabel('Avaliação Média')
        axes[0].set_xscale('log')
        axes[0].grid(True)

        # scatterplor do preço com desconto X avaliações
        sns.scatterplot(ax=axes[1], data=data, x='discounted_price', y='rating', size='rating_count', alpha=0.5)
        axes[1].set_title('Preço com Desconto X Avaliações')
        axes[1].set_xlabel('Preço com Desconto')
        axes[1].set_ylabel('Avaliação Média')
        axes[1].set_xscale('log')
        axes[1].grid(True)

        return plt.show()
    
    
    def sentiment_analysis(self, text):
        '''
        Método que cria valores para definir sentimentos
        '''
        
        analysis = TextBlob(text)
        
        if analysis.sentiment.polarity > 0.1:
            return 'Positive'
        elif analysis.sentiment.polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
        


    def prep_model(self, data2):
        data2['combined_text'] = data2['product_name'] + ' ' + data2['category'] + ' ' + data2['about_product'] + ' ' + data2['review_content']

        # fillna com strings vazias para evitar problemas
        data2['combined_text'] = data2['combined_text'].fillna('')


        vectorizer = TfidfVectorizer(stop_words='english', max_df = 0.95, min_df = 2, ngram_range=(1,1))

        # fitando e transformando
        tfidf_matrix = vectorizer.fit_transform(data2['combined_text'])
        
        return tfidf_matrix
    
    
    def hybrid_recommendation(self, product_id, content_sim_matrix, product_user_matrix, products, top_n=10):
        idx = products.index[products['product_id'] == product_id][0]
        
        # similaridade entre pares
        sim_scores = list(enumerate(content_sim_matrix[idx]))
        
        # classificando produtos com base na similaridade
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        conten_recommendation_idx = [i[0] for i in sim_scores[1:top_n+1]]
        
        
        # pegando as avaliações dos produtos
        if product_id in product_user_matrix.index:
            current_product_rating = product_user_matrix.loc[product_id].values[0]
            
        similar_rating_products = product_user_matrix.iloc[(product_user_matrix['rating']-current_product_rating).abs().argsort()[:top_n]]
        
        # combinando conteúdo de recomendação
        collaborative_recommendations_idx = similar_rating_products.index
        
        # mapeando indices dos produtos
        collaborative_recommendations_idx = [products.index[products['product_id'] == pid].tolist()[0] for pid in collaborative_recommendations_idx]
        
        # combinndo indices de ambos os métodos e removendo duplicados
        combined_indices = list(set(conten_recommendation_idx + collaborative_recommendations_idx))
        
        # obtendo detalhes dos produtos recomendados
        recommended_products = products.iloc[combined_indices].copy()
        recommended_products = recommended_products[['product_id', 'product_name', 'rating']]
        
        return recommended_products