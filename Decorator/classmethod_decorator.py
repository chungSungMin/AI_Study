'''
@classmethod decorator의 경우 class의 instance를 생성하지 않아도 바로 사용 가능하도록 해줍니다.
'''

class User:
    # 클래스 변수
    user_count = 0
    
    def __init__(self, name):
        # 인스턴스 변수
        self.name = name
        User.user_count += 1  # 클래스 변수를 수정

    @classmethod
    def print_user_count(cls):
        print(f"Total users: {cls.user_count}")

# 인스턴스 생성
user1 = User("Alice")
user2 = User("Bob")

# 인스턴스 변수는 각 인스턴스에 따라 다름
print(user1.name)  # 출력: Alice
print(user2.name)  # 출력: Bob

# 클래스 메서드는 인스턴스를 생성하지 않고 호출 가능
User.print_user_count()  # 출력: Total users: 2