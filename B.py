from collections import deque
import heapq

class Animal:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.order = 0

    def is_order_than(self, animal):
        return self.order < animal.order

class Dog(Animal):
    def __repr__(self):
        return "Dog named {0} (age {1})".format(self.name, self.age)

class Cat(Animal):
    def __repr__(self):
        return "Cat named {0} (age {1})".format(self.name, self.age)

class AnimalShelter:
    def add_animal(self, animal):
        raise NotImplementedError

    def get_dog(self):
        raise NotImplementedError

    def get_cat(self):
        raise NotImplementedError

    def get_any(self):
        raise NotImplementedError

class ArrivalBasedAnimalShelter(AnimalShelter):
    def __init__(self):
        self.dogs = deque()
        self.cats = deque()
        self.order = 0

    def add_animal(self, animal):
        animal.order = self.order
        self.order += 1
        if isinstance(animal, Dog):
            self.dogs.append(animal)
        elif isinstance(animal, Cat):
            self.cats.append(animal)

    def get_dog(self):
        return self.dogs.popleft() if self.dogs else None

    def get_cat(self):
        return self.cats.popleft() if self.cats else None

    def get_any(self):
        if not self.dogs:
            return self.get_cat()
        if not self.cats:
            return self.get_dog()

        if self.dogs[0].is_order_than(self.cats[0]):
            return self.get_dog()
        else:
            return self.get_cat()

class AgeBasedAnimalShelter(AnimalShelter):
    def __init__(self):
        self.dogs = []
        self.cats = []
        self.order = 0

    def add_animal(self, animal):
        animal.order = self.order
        self.order += 1
        if isinstance(animal, Dog):
            heapq.heappush(self.dogs, (-animal.age, animal.order, animal))
        elif isinstance(animal, Cat):
            heapq.heappush(self.cats, (-animal.age, animal.order, animal))

    def get_dog(self):
        return heapq.heappop(self.dogs)[-1] if self.dogs else None

    def get_cat(self):
        return heapq.heappop(self.cats)[-1] if self.cats else None

    def get_any(self):
        if not self.dogs:
            return self.get_cat()
        if not self.cats:
            return self.get_dog()

        dog = self.dogs[0]
        cat = self.cats[0]

        print(dog)
        print(cat)
        if (-dog[0], dog[1]) < (-cat[0], cat[1]):
            return self.get_dog()
        else:
            return self.get_cat()

def test_shelters():
    # Testing ArrivalBasedAnimalShelter
    arrival_shelter = ArrivalBasedAnimalShelter()

    dog1 = Dog("Buddy", 5)
    cat1 = Cat("Whiskers", 3)
    dog2 = Dog("Rex", 7)
    cat2 = Cat("Mittens", 2)

    arrival_shelter.add_animal(dog1)
    arrival_shelter.add_animal(cat1)
    arrival_shelter.add_animal(dog2)
    arrival_shelter.add_animal(cat2)

    assert arrival_shelter.get_any() == dog1
    assert arrival_shelter.get_dog() == dog2
    assert arrival_shelter.get_cat() == cat1
    assert arrival_shelter.get_any() == cat2

    # Testing AgeBasedAnimalShelter
    age_shelter = AgeBasedAnimalShelter()

    dog3 = Dog("Bruno", 6)#1
    cat3 = Cat("Kitty", 4)#2
    dog4 = Dog("Max", 4)#3
    cat4 = Cat("Luna", 6)#4

    age_shelter.add_animal(dog3)
    age_shelter.add_animal(cat3)
    age_shelter.add_animal(dog4)
    age_shelter.add_animal(cat4)

    assert age_shelter.get_any() == dog3  # Because Luna is the oldest at 6 years.
    assert age_shelter.get_dog() == dog4  # Because Bruno is older than Max.
    assert age_shelter.get_cat() == cat4
    assert age_shelter.get_any() == cat3

    print("All tests passed!")

test_shelters()
