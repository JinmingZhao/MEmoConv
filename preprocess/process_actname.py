'''
将文本中的
'''   
from FileOps import read_file

if __name__ == '__main__':    
    movies_names = read_file('../preprocess/movie_list.txt')
    movies_names = [movie_name.strip() for movie_name in movies_names]
    if True:
        # process replace the Person name in each movie. 
        # 根据每部电影的spk的名字，进行替换成 PERSON 这个token.
        for movie_name in movies_names:
            print(f'Current movie {movie_name}')