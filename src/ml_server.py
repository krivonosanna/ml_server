import os
import pickle
import numpy as np
import ensembles
import pandas as pd
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect

from wtforms import validators
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms.validators import DataRequired, NumberRange, Optional
from wtforms import StringField, SubmitField, SelectField, IntegerField, FloatField


app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
Bootstrap(app)


class NameForm(FlaskForm):
    name = SelectField('Выберите модель', choices=['RandomForest', 'GradientBoosting'], validators=[DataRequired()])
    submit = SubmitField('Выбрать')


class DateForest(FlaskForm):
    n_estimators = IntegerField('Введите число деревьев', validators=[DataRequired()])
    max_depth = IntegerField('Введите максимальную глубину дерева(необязательно)',
                             validators=[Optional()])
    feature = FloatField('Введите долю (от 0 до 1)  признаков для каждого дерева (необязательно)',
                         validators=[Optional()])
    data_train = FileField('Загрузите файл для обучения в формате csv',
                           validators=[FileRequired(), FileAllowed(['csv'], 'Неверный формат')])
    data_val = FileField('Загрузите файл для валидации в формате csv(необязательно)',
                          validators=[FileAllowed(['csv'], 'Неверный формат'), Optional()])
    target = StringField('Название столбца с целевой переменной', validators=[DataRequired()])

    submit = SubmitField('Обучить модель')



class DateForestAnsw(FlaskForm):
    n_estimators = IntegerField('n_estimators', validators=[DataRequired()])
    max_depth = IntegerField('max_depth)', validators=[Optional()])
    feature = FloatField('feature_subsample_size',
                         validators=[Optional()])
    data_train = FileField('Обучающаяся выбока',
                           validators=[FileRequired(), FileAllowed(['csv'], 'Неверный формат')])
    data_val = FileField('Валидационнная выборка',
                          validators=[FileAllowed(['csv'], 'Неверный формат'), Optional()])


class DateBoosting(FlaskForm):
    n_estimators = IntegerField('Введите число деревьев', validators=[DataRequired()])
    max_depth = IntegerField('Введите максимальную глубину дерева(необязательно)')
    feature = FloatField('Введите долю (от 0 до 1)  признаков для каждого дерева (необязательно)',
                         validators=[(NumberRange(0, 1, 'Число вне диапазона'))])
    learn = FloatField('Введите темп обучения - learning_rate (необязательно)')
    data_train = FileField('Загрузите файл для обучения в формате csv',
                           validators=[FileRequired(), FileAllowed(['csv'], 'Неверный формат')])
    data_val = FileField('Загрузите файл для валидации в формате csv(необязательно)',
                         validators=[FileRequired(), FileAllowed(['csv'], 'Неверный формат')])
    target = StringField('Название столбца с целевой переменной', validators=[DataRequired()])

    submit = SubmitField('Обучить модель')


class Info(FlaskForm):
    submit = SubmitField('Получить информацию о модели')


class Test(FlaskForm):
    submit = SubmitField('Выполнить предсказания')


class TestDate(FlaskForm):
    data_val = FileField('Загрузите файл для предсказания в формате csv',
                           validators=[FileRequired(), FileAllowed(['csv'], 'Неверный формат')])
    submit = SubmitField('Выполнить предсказания')


class ScoreVal(FlaskForm):
    submit = SubmitField('Для валлидационной выборки')


class ScoreTrain(FlaskForm):
    submit = SubmitField('Для обучающейся выборки')


class Bac(FlaskForm):
    submit = SubmitField('Назад')


mod = ensembles.RandomForestMSE(1)


@app.route('/sentiment', methods=['GET', 'POST'])
def model():
    try:
        name_form = NameForm()

        if name_form.validate_on_submit():
            app.logger.info('On text: {0}'.format(name_form.name.data))
            if name_form.name.data == 'RandomForest':
                return redirect(url_for('forest'))
            else:
                return redirect(url_for('boosting'))
        return render_template('name_form.html', form=name_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/forest', methods=['GET', 'POST'])
def forest():
    try:
        d = DateForest()

        if d.validate_on_submit():
            app.logger.info('On t: {0}'.format(d.data_train.data))
            mod = ensembles.RandomForestMSE(d.n_estimators.data, d.max_depth.data, d.feature.data)
            f = d.data_train.data
            train = pd.read_csv(f)
            X = np.array(train.drop([d.target.data], axis=1))
            y = np.array(train[d.target.data])
            app.logger.info('On t: {0}'.format(d.data_val.data))
            if d.data_val.data is not None:
                test = pd.read_csv(d.data_val.data)
                y_val = np.array(test[d.target.data])
                X_val = np.array(test.drop([d.target.data], axis=1))
            else:
                X_val = None
                y_val = None
            app.logger.info('On t: {0}'.format(d.data_val.data))
            if X_val is not None:
                score_val, time_val, score_train = mod.fit(X, y, X_val, y_val, True)
            else:
                score_val = None
                score_train = mod.fit(X, y, X_val, y_val, True)
            app.logger.info('On t: {0}'.format(d))
            #return redirect(url_for('result'))
            return redirect(url_for('choice', score_val=score_val, score_train=score_train,
                                    n_estimators=d.n_estimators.data, max_depth=d.max_depth.data,
                                    feature=d.feature.data))

        return render_template('form_forest.html', form=d)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/boosting', methods=['GET', 'POST'])
def boosting():
    try:
        d = DateBoosting()

        if d.validate_on_submit():
            app.logger.info('On t: {0}'.format(d.data_train.data))
            mod = ensembles.RandomForestMSE(d.n_estimators.data, d.learn.data, d.max_depth.data, d.feature.data)
            f = d.data_train.data
            train = pd.read_csv(f)
            X = np.array(train.drop([d.target.data, 'date'], axis=1))
            y = np.array(train[d.target.data])
            app.logger.info('On t: {0}'.format(d.data_val.data))
            if d.data_val.data is not None:
                test = pd.read_csv(d.data_val.data)
                y_val = np.array(test[d.target.data])
                X_val = np.array(test.drop([d.target.data, 'date'], axis=1))
            else:
                X_val = None
                y_val = None
            app.logger.info('On t: {0}'.format(d.data_val.data))
            if X_val is not None:
                score_val, time_val, score_train = mod.fit(X, y, X_val, y_val, True)
            else:
                score_val = None
                score_train = mod.fit(X, y, X_val, y_val, True)
            app.logger.info('On t: {0}'.format(d))
            # return redirect(url_for('result'))
            return redirect(url_for('choice_2', score_val=score_val, score_train=score_train,
                                    n_estimators=d.n_estimators.data, max_depth=d.max_depth.data,
                                    feature=d.feature.data, learn=d.learn.data))

        return render_template('form_boosting.html', form=d)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/choice', methods=['GET', 'POST'])
def choice():
    try:
        score_val = request.args.get('score_val')
        score_train = request.args.get('score_train')
        n_estimators = request.args.get('n_estimators')
        max_depth = request.args.get('max_depth')
        feature = request.args.get('feature')
        test = Test()
        if test.validate_on_submit():
            app.logger.info('On t: {0}'.format(mod))
            return redirect(url_for('test'))
        return render_template('result.html', n_estimators=n_estimators, max_depth=max_depth,
                               feature=feature, score_train=score_train, score_val=score_val, form=test)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/choicebust', methods=['GET', 'POST'])
def choice_2():
    try:
        score_val = request.args.get('score_val')
        score_train = request.args.get('score_train')
        n_estimators = request.args.get('n_estimators')
        max_depth = request.args.get('max_depth')
        feature = request.args.get('feature')
        learn = request.args.get('learn')
        test = Test()
        if test.validate_on_submit():
            return redirect(url_for('test'))
        return render_template('result_bust.html', n_estimators=n_estimators, max_depth=max_depth, learn=learn,
                               feature=feature, score_train=score_train, score_val=score_val, form=test)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/test', methods=['GET', 'POST'])
def test():
    form = TestDate()
    y_pred = request.args.get('y')
    if form.validate_on_submit():
        f = form.data_val.data
        X = pd.read_csv(f)
        y_pred = mod.predict(X)
        return redirect(url_for('test', y=y_pred))
    return render_template('test.html', form=form, y=y_pred)


@app.errorhandler(404)
def handle_404(e):
    return redirect(url_for('model'))