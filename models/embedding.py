import torch
import typing as tp
from sentence_transformers import SentenceTransformer


def create_embedding(MODEL_NAME='all-MiniLM-L6-v2'):
    '''
    :return: returns callable function(data.x) -> torch.Tensor[num_nodes, EMBEDDING_SIZE]
    '''
    model = SentenceTransformer('sentence-transformers/' + MODEL_NAME,
                                device='cuda' if torch.cuda.is_available() else 'cpu')
    EMBEDDING_SIZE = model.get_sentence_embedding_dimension()
    embeddings = dict()

    def encode_token(token: str):
        if token not in embeddings.keys():
            embeddings[token] = model.encode(token)
        #             print(f'encoded new token: {token}')
        return embeddings[token]

    def f(args: tp.List[tp.List[tp.List]] = []):
        # args =  [[['SimpleType', 'SimpleName'], ['SimpleName'], ['SingleVariableDeclaration']], ...]
        # :return: torch.Tensor[len(args), EMBEDDING_SIZE]
        embs_0 = torch.zeros([len(args), EMBEDDING_SIZE], dtype=torch.float)

        for i, arg_1 in enumerate(args):
            # node = [['SimpleType', 'SimpleName'], ['SimpleName'], ['SingleVariableDeclaration']]
            embs_1 = torch.zeros(EMBEDDING_SIZE)

            if isinstance(arg_1, str):
                embs_0[i] = encode_token(arg_1)
                continue

            for arg_2 in arg_1:  # arg = ['SimpleType', 'SimpleName']
                embs_2 = torch.zeros(EMBEDDING_SIZE)
                if isinstance(arg_2, str):
                    embs_1 += encode_token(arg_2)
                    continue

                for token in arg_2:  # token = 'SimpleType'
                    embs_2 += encode_token(token)
                embs_2 /= len(arg_2)
                embs_1 += embs_2

            embs_1 /= len(arg_1)
            embs_0[i] = embs_1

        return embs_0

    return f
