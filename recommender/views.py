from django.http import JsonResponse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import requests
import pandas as pd
import numpy as np

def fetch_data():
    api_url = 'http://localhost:8081/tache/get-taches'
    response = requests.get(api_url)
    if response.status_code == 200:
        data = pd.DataFrame(response.json())
        data['competences'] = data['competences'].apply(lambda x: ', '.join([tech['technologies'] for tech in x]))
        return data
    else:
        return pd.DataFrame()

def give_recommendations(request, task_id):
    data = fetch_data()
    if not data.empty:
        text_data = data['competences'].tolist()
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        embeddings = model.encode(text_data, show_progress_bar=True)
        X = np.array(embeddings)
        pca = PCA(n_components=len(data))
        pca.fit(X)
        pca_data = pd.DataFrame(pca.transform(X))
        cos_sim_data = pd.DataFrame(cosine_similarity(X))

        index = data[data['id'] == task_id].index[0]
        index_recomm = cos_sim_data.loc[index].sort_values(ascending=False).index.tolist()[1:len(data)]
        task_recom = data['id'].iloc[index_recomm].values
        result = {'Tasks': task_recom.tolist(), 'Index': index_recomm}
        return JsonResponse(result)
    else:
        return JsonResponse({'error': 'Error fetching data'})

