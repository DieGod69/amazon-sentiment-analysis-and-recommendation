import os
import warnings
import numpy as np
import pandas as pd
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