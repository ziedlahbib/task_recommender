# from django.http import JsonResponse
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.decomposition import PCA
# import requests
# import pandas as pd
# import numpy as np

# def fetch_data():
#     api_url = 'http://localhost:8081/tache/get-taches'
#     response = requests.get(api_url)
#     if response.status_code == 200:
#         data = pd.DataFrame(response.json())
#         data['competences'] = data['competences'].apply(lambda x: ', '.join([tech['technologies'] for tech in x]))
#         return data
#     else:
#         return pd.DataFrame()

# def give_recommendations(request, task_id):
#     data = fetch_data()
#     if not data.empty:
#         text_data = data['competences'].tolist()
#         model = SentenceTransformer('distilbert-base-nli-mean-tokens')
#         embeddings = model.encode(text_data, show_progress_bar=True)
#         X = np.array(embeddings)
#         pca = PCA(n_components=len(data))
#         pca.fit(X)
#         pca_data = pd.DataFrame(pca.transform(X))
#         cos_sim_data = pd.DataFrame(cosine_similarity(X))

#         index = data[data['id'] == task_id].index[0]
#         index_recomm = cos_sim_data.loc[index].sort_values(ascending=False).index.tolist()[1:len(data)]
#         task_recom = data['id'].iloc[index_recomm].values
#         result = {'Tasks': task_recom.tolist(), 'Index': index_recomm}
#         return JsonResponse(result)
#     else:
#         return JsonResponse({'error': 'Error fetching data'})

from django.http import JsonResponse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import requests
import pandas as pd
import numpy as np

def fetch_data(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        data = pd.DataFrame(response.json())
        return data
    else:
        return pd.DataFrame()

def prepare_data():
    # Fetch task data
    tasks_data = fetch_data('http://localhost:8081/tache/get-taches')
    if not tasks_data.empty:
        tasks_data['competences'] = tasks_data['competences'].apply(
            lambda x: ', '.join([tech['technologies'] for tech in x])
        )
    
    # Fetch user data
    users_data = fetch_data('http://localhost:8081/user/get-users')
    if not users_data.empty:
        users_data['competences'] = users_data['userCompetences'].apply(
            lambda x: ', '.join([tech['competence']['technologies'] for tech in x])
        )
    
    return tasks_data, users_data

def give_recommendations(request, task_id):
    tasks_data, users_data = prepare_data()
    if not tasks_data.empty and not users_data.empty:
        task_text_data = tasks_data['competences'].tolist()
        user_text_data = users_data['competences'].tolist()
        
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')

        # Encode task and user competencies
        task_embeddings = model.encode(task_text_data, show_progress_bar=True)
        user_embeddings = model.encode(user_text_data, show_progress_bar=True)

        # Apply PCA to reduce dimensionality
        # pca_tasks = PCA(n_components=min(len(task_embeddings), len(users_data)))
        # pca_users = PCA(n_components=min(len(user_embeddings), len(tasks_data)))
        # task_embeddings = pca_tasks.fit_transform(task_embeddings)
        # user_embeddings = pca_users.fit_transform(user_embeddings)

        cos_sim_data = cosine_similarity(task_embeddings, user_embeddings)

        task_index = tasks_data[tasks_data['id'] == task_id].index[0]
        similarities = cos_sim_data[task_index]
        top_user_indices = similarities.argsort()[-len(user_text_data):][::-1]  # Top 5 users

        recommended_users = users_data.iloc[top_user_indices]

        result = {
            'TaskID': task_id,
            'RecommendedUsers': recommended_users[['id', 'competences']].to_dict(orient='records')
        }
        print(result)
        return JsonResponse(result)
    else:
        return JsonResponse({'error': 'Error fetching data'})
