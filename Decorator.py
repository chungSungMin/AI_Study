# 간단하게 User 설정하는 Class를 구현한다.
class User:
    def __init__(self, is_admin):
        self.is_admin = is_admin

# 해당 User가 admin인지 guest인지 판별하기 위한 수단.

# requires_admin은 데코레이팅 함수 ( delete_database ) 를 인자로 받아옵니다
def requires_admin(func):
    # wapper 함수 구현
    def check_admin(user, *args, **kwargs):
        # 만일 권한이 없다면 delete_database를 실행하지 않고 에러를 발생시킵니다.
        if not user.is_admin :
            raise PermissionError('권한이 필요합니다.')
        # 원래 함수를 실행시킵니다 ( *args, **kwargs는 확장을 위해 필수적으로 넣는 인자들입니다 )
        return func(user, *args, **kwargs)
    return check_admin
    
# 데코레이터는 원래 함수를 변경하거나 확장하게 됩니다.
@requires_admin
def delete_database(user):
    print('Database를 삭제하겠습니다.')



print("User1 접근")
user1 = User(is_admin=True)
delete_database(user1)

print("User2 접근")
user2 = User(is_admin=False)
delete_database(user2)