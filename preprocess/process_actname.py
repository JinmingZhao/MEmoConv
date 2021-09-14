'''
将每部电影中出现的人名统一替换为 PERSON, 在送入Bert的时候可以用 [uncased88] special token 替换.
'''   
from FileOps import read_file, read_xls, read_csv, write_csv

if __name__ == '__main__':    
    movies_names = read_file('../preprocess/movie_list_total.txt')
    movies_names = [movie_name.strip() for movie_name in movies_names]
    dialog_spk_info_filepath = '/Users/jinming/Desktop/works/memoconv_final_labels/dialogSpkAnnoUpdate.xlsx'

    if True:
        # process replace the Person name in each movie. 
        # 根据每部电影的spk的名字，进行替换成 PERSON 这个token.
        for movie_name in movies_names:
            print(f'Current movie {movie_name}')
            meta_fileapth = '/Users/jinming/Desktop/works/memoconv_final_labels_csv/meta_{}.csv'.format(movie_name)
            new_meta_fileapth = '/Users/jinming/Desktop/works/memoconv_final_labels_csv_personname/meta_{}.csv'.format(movie_name)
            all_instances = read_xls(dialog_spk_info_filepath, sheetname=movie_name, skip_rows=0)
            movie_act_names = []
            actor_names = all_instances[0][1:]
            other_names = all_instances[3][1:]
            for actor, other in zip(actor_names, other_names):
                actor_name, others = actor.value, other.value
                if actor_name is not None:
                    movie_act_names.append(actor_name)
                if others is not None:
                    sep_node = ' '
                    if '；' in others:
                        sep_node = '；'
                    if '，' in others:
                        sep_node = '，'
                    other_names = others.split(sep_node)
                    movie_act_names.extend(other_names)
            # 比如 李子维(Adult) 只保留前面的部分
            new_movie_act_names = []
            for m_name in movie_act_names:
                if '(' in m_name:
                    new_movie_act_names.append(m_name[:m_name.index('(')])
                elif '（' in m_name:
                    new_movie_act_names.append(m_name[:m_name.index('（')])
                else:
                    new_movie_act_names.append(m_name)
            print(new_movie_act_names)
            new_instances = []
            all_instances = read_csv(meta_fileapth, delimiter=';', skip_rows=1)
            new_instances.append(all_instances[0])
            for instance in all_instances:
                _,_,_,text = instance[:4]
                for m_name in new_movie_act_names:
                    if m_name in text:
                        text = text.replace(m_name, '[unused88]')
                instance[3] = text
                new_instances.append(instance)
            write_csv(new_meta_fileapth, new_instances, delimiter=';')