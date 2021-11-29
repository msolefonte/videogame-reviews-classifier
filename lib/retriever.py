import requests
from bs4 import BeautifulSoup


def get_critic_reviews_from_game(game_id):
    game_ratings = []
    game_reviews = []

    url = 'https://www.metacritic.com' + game_id + '/critic-reviews'
    user_agent = {'User-agent': 'Mozilla/5.0'}

    response = requests.get(url, headers=user_agent)
    beautiful_soup = BeautifulSoup(response.text, 'html.parser')

    for review in beautiful_soup.find('div', class_='critic_reviews_module').find_all('div', class_='review_content'):
        if review.find('span', class_='blurb blurb_expanded'):
            game_reviews.append(review.find(
                'span', class_='blurb blurb_expanded').text)
            game_ratings.append(review.find(
                'div', class_='review_grade').find_all('div')[0].text)
        else:
            try:
                try:
                    game_reviews.append(review.find(
                        'div', class_='review_body').find('span').text)
                except AttributeError:
                    game_reviews.append(review.find(
                        'div', class_='review_body').text)
                game_ratings.append(review.find(
                    'div', class_='review_grade').find_all('div')[0].text)
            except AttributeError:
                continue

    return game_ratings, game_reviews


def get_latest_games():
    game_ids = []

    i = 0
    while True:
        url = 'https://www.metacritic.com/browse/games/score/userscore/90day/all/filtered?sort=desc&page='
        user_agent = {'User-agent': 'Mozilla/5.0'}

        response = requests.get(url + str(i), headers=user_agent)
        beautiful_soup = BeautifulSoup(response.text, 'html.parser')

        if len(beautiful_soup.find_all('div', class_='clamp-score-wrap')) == 0:
            break

        for game in beautiful_soup.find_all('div', class_='clamp-score-wrap'):
            game_id = game.find('a', class_='metascore_anchor')[
                'href'].rsplit('/', 1)[0]
            game_ids.append(game_id)
        i += 1

    return game_ids


def main():
    all_game_ratings = []
    all_game_reviews = []

    game_ids = get_latest_games()
    for game_id in game_ids:
        game_ratings, game_reviews = get_critic_reviews_from_game(game_id)

        all_game_ratings += game_ratings
        all_game_reviews += game_reviews

    print('rating\treview')
    for i in range(len(all_game_reviews)):
        if all_game_ratings[i].isdigit():
            print(all_game_ratings[i] + '\t' +
                  ''.join(all_game_reviews[i].splitlines()))


if __name__ == "__main__":
    main()
