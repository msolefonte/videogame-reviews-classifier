import requests
from bs4 import BeautifulSoup


def get_user_reviews_from_game(game_id):
    game_ratings = []
    game_reviews = []

    for page in range(0, 10):
        url = 'https://www.metacritic.com' + \
            game_id + '/user-reviews?page=' + str(page)
        user_agent = {'User-agent': 'Mozilla/5.0'}

        response = requests.get(url, headers=user_agent)
        beautiful_soup = BeautifulSoup(response.text, 'html.parser')

        for review in beautiful_soup.find_all('div', class_='review_content'):
            if review.find('div', class_='name') is None:
                break
            if review.find('span', class_='blurb blurb_expanded'):
                game_reviews.append(review.find(
                    'span', class_='blurb blurb_expanded').text)
                game_ratings.append(review.find(
                    'div', class_='review_grade').find_all('div')[0].text)
            else:
                try:
                    game_reviews.append(review.find(
                        'div', class_='review_body').find('span').text)
                    game_ratings.append(review.find(
                        'div', class_='review_grade').find_all('div')[0].text)
                except AttributeError:
                    continue

    return game_ratings, game_reviews


def get_best_games_from_year(year):
    url = 'https://www.metacritic.com/browse/games/score/metascore/year/filtered?year_selected=' + \
        str(year)
    user_agent = {'User-agent': 'Mozilla/5.0'}

    response = requests.get(url, headers=user_agent)
    beautiful_soup = BeautifulSoup(response.text, 'html.parser')

    game_ids = []
    for game in beautiful_soup.find_all('div', class_='clamp-score-wrap'):
        game_id = game.find('a', class_='metascore_anchor')[
            'href'].rsplit('/', 1)[0]
        game_ids.append(game_id)

    return game_ids


def main():
    all_game_ratings = []
    all_game_reviews = []

    game_ids = get_best_games_from_year(2020)
    for game_id in game_ids:
        game_ratings, game_reviews = get_user_reviews_from_game(game_id)

        all_game_ratings += game_ratings
        all_game_reviews += game_reviews

    print('rating\treview')
    for i in range(len(all_game_reviews)):
        print(all_game_ratings[i] + '\t' +
              ''.join(all_game_reviews[i].splitlines()))


if __name__ == "__main__":
    main()
