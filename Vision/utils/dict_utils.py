from collections import defaultdict

def invert_map(class_map):
    """
    Devuelve el mapa inverso con valores por clave y viceversa
    :param class_map: Mapa a invertir
    :return: Mapa invertido
    """
    inv_map = dict()
    for k, v in class_map.items():
        inv_map[v] = k
    return inv_map

class DefaultDict(defaultdict):
    """
    Diccionario con funcion para devolver para devolver valores para
    claves faltantes. Lor valores resultantes son el resultado de
    aplicar la funcion factoria a la clave faltante.
    """
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError((key,))
        value = self.default_factory(key)
        return value


class DefaultDictAddMissing(defaultdict):
    """
    Diccionario con funcion para devolver para devolver valores para
    claves faltantes. Lor valores resultantes son el resultado de
    aplicar la funcion factoria a la clave faltante.

    Ademas, una vez se calcula el valor de la funcion la guardamos
    en el diccionario
    """
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError((key,))
        self[key] = value = self.default_factory(key)
        return value
