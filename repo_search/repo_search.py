print("Loading libraries...")
import openai
#import faiss
import tqdm
import os
import sys
import zipfile
import urllib.request
import tiktoken
import datasets
import time
import numpy as np
from io import BytesIO

# Module constants

# Map from url to what we need to append to get the zip file
supported_remote_repositories = {
    'https://github.com': "/archive/refs/heads/main.zip",
    #'https://gitlab.com': "/-/archive/main/REPONAME-main.zip", # Missing repository name, which would be difficult to get into here.
    'https://bitbucket.org': "/get/main.zip",
}

# OpenAI API key
#OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")

OPENAI_MODEL = 'text-embedding-ada-002'

OPENAI_MODEL_MAX_INPUT_TOKENS = 8191

openai_encoder = tiktoken.get_encoding("cl100k_base")

# Exported functions

def generate_embeddings_for_repository(
    dataset_name,
    repo_url_or_path,
    embeddings_dir,
    verbose):
    if dataset_exists(dataset_name, embeddings_dir):
        # To help the user, give the full disk path to the embeddings directory
        embeddings_dir_expanded = os.path.abspath(embeddings_dir)

        print(f'Dataset named {dataset_name} already exists in embeddings directory ({embeddings_dir_expanded}), delete it first if you want to regenerate it.')
        return
    
    # Check if given repository is a URL or a local path
    if repo_url_or_path.startswith('http'):
        # Remote repositories
        if repo_url_or_path.endswith('.zip'):
            generate_embeddings_for_remote_zip_archive(dataset_name, repo_url_or_path, embeddings_dir, verbose)
        elif is_supported_remote_repository(repo_url_or_path):
            generate_embeddings_for_remote_repository_archive(dataset_name, repo_url_or_path, embeddings_dir, verbose)
        else:
            print(f'ERROR: Unsupported remote repository: {repo_url_or_path}')
            return
    else:
        # Local repositories
        if repo_url_or_path.startswith('~'):
            repo_url_or_path = os.path.expanduser(repo_url_or_path)
        
        if os.path.isdir(repo_url_or_path):
            generate_embeddings_for_local_repository(dataset_name, repo_url_or_path, embeddings_dir, verbose)
        elif repo_url_or_path.endswith('.zip'):
            generate_embeddings_for_local_zip_archive(dataset_name, repo_url_or_path, embeddings_dir, verbose)
        else:
            print(f'ERROR: Unsupported local repository: {repo_url_or_path}')
            return



def query_embeddings(
    dataset_name,
    query,
    embeddings_dir,
    verbose):
    if not dataset_exists(dataset_name, embeddings_dir):
        # To help the user, give the full disk path to the embeddings directory
        embeddings_dir_expanded = os.path.abspath(embeddings_dir)

        print(f'Dataset named {dataset_name} does not exist in embeddings directory ({embeddings_dir_expanded}), generate it first.')
        return
    
    print('Querying embeddings...')

    query_embedding = generate_embedding_for_chunk(query, verbose)

    # Load the dataset from disk.
    dataset = datasets.load_from_disk(os.path.join(embeddings_dir, dataset_name))

    # Load the index from disk.
    # TODO

    embeddings = dataset['embeddings']
    similarities = []

    for embedding_list in tqdm.tqdm(embeddings):
        # Each file may have more than one embedding
        # Check them all to see which one is most similar to the query
        best_similarity = 0.0
        for embedding in embedding_list:
            similarity = cosine_similarity(query_embedding, embedding)
            if similarity > best_similarity:
                best_similarity = similarity
        similarities.append(best_similarity)
    
    # Sort the similarities and file paths in descending order.
    similarities, file_paths = zip(*sorted(zip(similarities, dataset['file_path']), reverse=True))

    # Print the top 10 results.
    for similarity, file_path in zip(similarities[:10], file_paths[:10]):
        print(f'{similarity}: {file_path}')


# Internal functions
def dataset_exists(dataset_name, embeddings_dir):
    # Check if folder named dataset_name exists in embeddings_dir.
    return os.path.exists(os.path.join(embeddings_dir, dataset_name))

def is_supported_remote_repository(repo_url):
    return any(
        repo_url.startswith(supported_url)
        for supported_url in supported_remote_repositories
    )

def get_download_url_for_remote_repository(repo_url):
    return next(
        (
            repo_url + supported_remote_repositories[supported_url]
            for supported_url in supported_remote_repositories
            if repo_url.startswith(supported_url)
        ),
        None,
    )


## Generator functions
def generate_embeddings_for_remote_repository_archive(
    dataset_name,
    repo_url,
    embeddings_dir,
    verbose):
    assert is_supported_remote_repository(repo_url)

    for supported_url in supported_remote_repositories:
        if repo_url.startswith(supported_url):
            download_url = repo_url + supported_remote_repositories[supported_url]
            break
    else:
        print(f'ERROR: Unsupported remote repository: {repo_url}')
        return

    if verbose:
        print(f'Detected {supported_url} repository.')

    generate_embeddings_for_remote_zip_archive(
        dataset_name,
        download_url,
        embeddings_dir,
        verbose
    )


def generate_embeddings_for_remote_zip_archive(
    dataset_name,
    zip_url,
    embeddings_dir,
    verbose):
    # Use zipfile to browse the contents of the zip file without extracting it.
    with BytesIO() as zip_buffer:
        if verbose:
            print(f'Downloading {zip_url}...')

        with urllib.request.urlopen(zip_url) as response:
            zip_buffer.write(response.read())
            zip_buffer.seek(0)

        with zipfile.ZipFile(zip_buffer) as zip_ref:
            generate_embeddings_for_zipfile(
                dataset_name,
                zip_ref,
                embeddings_dir,
                verbose
            )

def generate_embeddings_for_local_zip_archive(
    dataset_name,
    zip_path,
    embeddings_dir,
    verbose):
    if verbose:
        print(f'Loading {zip_path}...')
    
    # Use zipfile to browse the contents of the zip file without extracting it.
    with zipfile.ZipFile(zip_path) as zip_ref:
        generate_embeddings_for_zipfile(
            dataset_name,
            zip_ref,
            embeddings_dir,
            verbose
        )

def generate_embeddings_for_zipfile(
    dataset_name,
    zipfile,
    embeddings_dir,
    verbose):

    if verbose:
        print(f'Generating embeddings from zipfile for {dataset_name}...')
    
    file_list = zipfile.namelist()

    # For each file in the zip file, generate embeddings for it.
    all_embeddings = []
    for file_path in tqdm.tqdm(file_list):
        try:
            with zipfile.open(file_path, 'r') as file:
                file_contents = file.read()
                file_contents = file_contents.decode('utf-8')
                all_embeddings.append(generate_embeddings_for_contents(file_contents, verbose))
        except UnicodeDecodeError as e:
            if verbose:
                print(f'WARNING: Could not read as text file: {file_path}')
            all_embeddings.append([])
        #except:
        #    if verbose:
        #        print(f'WARNING: Issue generating embeddings for: {file_path}')
        #    all_embeddings.append([])
    
    # Generate a dataset from the embeddings.
    dataset = datasets.Dataset.from_dict({
        'file_path': file_list,
        'embeddings': all_embeddings
    })

    # Save the dataset to disk.
    dataset.save_to_disk(os.path.join(embeddings_dir, dataset_name))

    # Generate index using FAISS.
    generate_faiss_index_for_dataset(dataset, dataset_name, embeddings_dir, verbose)


def generate_embeddings_for_local_repository(
    dataset_name,
    repo_path,
    embeddings_dir,
    verbose):

    if verbose:
        print(f'Generating embeddings from local directory {repo_path} for {dataset_name}...')

    file_paths = []
    embeddings = []

    # Populate file_paths with all files in repo_path and its subdirectories.
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if the file can be read as a text file
            try:
                with open(os.path.join(root, file), 'rt') as file:
                    _ = file.read() # TODO: Is this needed? Does the exception get generated by open or read?
            except UnicodeDecodeError as e:
                if verbose:
                    print(f'WARNING: Could not read as text file: {file_path}')
                continue
            
            # When storing file_path, remove shared repo_path prefix.
            relative_file_path = file_path[len(repo_path) + 1:]
            file_paths.append(relative_file_path)
    
    # Generate embeddings for each file in file_paths.
    for file_path in tqdm.tqdm(file_paths):
        full_file_path = os.path.join(repo_path, file_path)
        try:
            with open(full_file_path, 'rt') as file:
                file_contents = file.read()
                embedding = generate_embeddings_for_contents(file_contents, verbose)
                embeddings.append(embedding)
        except:
            if verbose:
                print(f'WARNING: Issue generating embeddings for: {full_file_path}')
            embeddings.append([])
    
    # Generate a dataset from the embeddings.
    dataset = datasets.Dataset.from_dict({
        'file_path': file_paths,
        'embeddings': embeddings
    })

    # Save the dataset to disk.
    dataset.save_to_disk(os.path.join(embeddings_dir, dataset_name))

    # Generate index using FAISS.
    generate_faiss_index_for_dataset(dataset, dataset_name, embeddings_dir, verbose)

def generate_embeddings_for_contents(
    file_contents,
    verbose
):
    # Use tiktoken to split file_contents into chunks of OPENAI_MODEL_MAX_INPUT_TOKENS.
    tokens = openai_encoder.encode(file_contents)

    # Split tokens into chunks of OPENAI_MODEL_MAX_INPUT_TOKENS.
    all_embeddings = []
    for i in range(0, len(tokens), OPENAI_MODEL_MAX_INPUT_TOKENS):
        chunk = tokens[i:i + OPENAI_MODEL_MAX_INPUT_TOKENS]
        chunk = encoding.decode(chunk)
        all_embeddings.append(generate_embedding_for_chunk(chunk, verbose))
    
    return all_embeddings

def generate_embedding_for_chunk_FAKE(
    file_chunk,
    verbose
):
    # Debug testing
    return [0.0] * 1536

def generate_embedding_for_chunk(
    file_chunk,
    verbose,
):
    assert len(openai_encoder.encode(file_chunk)) <= OPENAI_MODEL_MAX_INPUT_TOKENS

    current_try = 0
    max_tries = 5

    while current_try <= max_tries:
        current_try += 1
        try:
            embedding_response = openai.Embedding.create(
                input=file_chunk,
                model=OPENAI_MODEL,
            )
            break
        except openai.error.OpenAIError as e:
            if verbose:
                print(f'WARNING: OpenAI API error: {e}')
            
            if current_try == max_tries:
                raise e
            
            # Exponential backoff
            time.sleep(2**current_try)

    return embedding_response['data'][0]['embedding']

def generate_faiss_index_for_dataset(
    dataset,
    dataset_name,
    embeddings_dir,
    verbose):
    pass

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))