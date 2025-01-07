import yaml


def load_yaml(yaml_dir):
    yaml_file=open(yaml_dir, 'r')
    data=yaml.safe_load(yaml_file)
    yaml_file.close()
    return data



def get_yaml_args(yaml_list):
    # print(yaml_list)
    # exit()
    yaml_out={}
    for a in yaml_list:
        a=a.split(' ') 
        yaml_out[a[0]]=load_yaml(a[1])

    
    return yaml_out