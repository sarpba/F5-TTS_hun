# -*- coding: utf-8 -*-
# Copyright (c) 2003, Taro Ogawa.  All Rights Reserved.
# Copyright (c) 2013, Savoir-faire Linux inc.  All Rights Reserved.

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301 USA

from __future__ import unicode_literals, division, print_function

import math
from collections import OrderedDict
from decimal import Decimal

# to_s függvény a compat.py-ból
def to_s(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)

# parse_currency_parts és prefix_currency függvények a currency.py-ból
def parse_currency_parts(val):
    is_negative = False
    if val < 0:
        is_negative = True
        val = abs(val)

    if isinstance(val, float):
        left, right = str(val).split('.')
        left = int(left)
        right = int(right)
    else:
        left = int(val)
        right = 0 # Egész számok esetén nincs tizedes rész

    return left, right, is_negative

def prefix_currency(prefix, currency_form):
    return (prefix + ' ' + currency_form[0], prefix + ' ' + currency_form[1])


class Num2Word_Base(object):
    CURRENCY_FORMS = {}
    CURRENCY_ADJECTIVES = {}

    def __init__(self):
        self.is_title = False
        self.precision = 2
        self.exclude_title = []
        self.negword = "(-) "
        self.pointword = "(.)"
        self.errmsg_nonnum = "type(%s) not in [long, int, float]"
        self.errmsg_floatord = "Cannot treat float %s as ordinal."
        self.errmsg_negord = "Cannot treat negative num %s as ordinal."
        self.errmsg_toobig = "abs(%s) must be less than %s."

        self.setup()

        # uses cards
        if any(hasattr(self, field) for field in
               ['high_numwords', 'mid_numwords', 'low_numwords']):
            self.cards = OrderedDict()
            self.set_numwords()
            self.MAXVAL = 1000 * list(self.cards.keys())[0]

    def set_numwords(self):
        self.set_high_numwords(self.high_numwords)
        self.set_mid_numwords(self.mid_numwords)
        self.set_low_numwords(self.low_numwords)

    def set_high_numwords(self, *args):
        raise NotImplementedError

    def set_mid_numwords(self, mid):
        for key, val in mid:
            self.cards[key] = val

    def set_low_numwords(self, numwords):
        for word, n in zip(numwords, range(len(numwords) - 1, -1, -1)):
            self.cards[n] = word

    def splitnum(self, value):
        for elem in self.cards:
            if elem > value:
                continue

            out = []
            if value == 0:
                div, mod = 1, 0
            else:
                div, mod = divmod(value, elem)

            if div == 1:
                out.append((self.cards[1], 1))
            else:
                if div == value:  # The system tallies, eg Roman Numerals
                    return [(div * self.cards[elem], div*elem)]
                out.append(self.splitnum(div))

            out.append((self.cards[elem], elem))

            if mod:
                out.append(self.splitnum(mod))

            return out

    def parse_minus(self, num_str):
        """Detach minus and return it as symbol with new num_str."""
        if num_str.startswith('-'):
            # Extra spacing to compensate if there is no minus.
            return '%s ' % self.negword.strip(), num_str[1:]
        return '', num_str

    def str_to_number(self, value):
        return Decimal(value)

    def to_cardinal(self, value):
        try:
            assert int(value) == value
        except (ValueError, TypeError, AssertionError):
            return self.to_cardinal_float(value)

        out = ""
        if value < 0:
            value = abs(value)
            out = "%s " % self.negword.strip()

        if value >= self.MAXVAL:
            raise OverflowError(self.errmsg_toobig % (value, self.MAXVAL))

        val = self.splitnum(value)
        words, num = self.clean(val)
        return self.title(out + words)

    def float2tuple(self, value):
        pre = int(value)

        # Simple way of finding decimal places to update the precision
        self.precision = abs(Decimal(str(value)).as_tuple().exponent)

        post = abs(value - pre) * 10**self.precision
        if abs(round(post) - post) < 0.01:
            # We generally floor all values beyond our precision (rather than
            # rounding), but in cases where we have something like 1.239999999,
            # which is probably due to python's handling of floats, we actually
            # want to consider it as 1.24 instead of 1.23
            post = int(round(post))
        else:
            post = int(math.floor(post))

        return pre, post

    def to_cardinal_float(self, value):
        try:
            float(value) == value
        except (ValueError, TypeError, AssertionError, AttributeError):
            raise TypeError(self.errmsg_nonnum % value)

        pre, post = self.float2tuple(float(value))

        post = str(post)
        post = '0' * (self.precision - len(post)) + post

        out = [self.to_cardinal(pre)]
        if value < 0 and pre == 0:
            out = [self.negword.strip()] + out

        if self.precision:
            out.append(self.title(self.pointword))

        for i in range(self.precision):
            curr = int(post[i])
            out.append(to_s(self.to_cardinal(curr)))

        return " ".join(out)

    def merge(self, curr, next):
        raise NotImplementedError

    def clean(self, val):
        out = val
        while len(val) != 1:
            out = []
            left, right = val[:2]
            if isinstance(left, tuple) and isinstance(right, tuple):
                out.append(self.merge(left, right))
                if val[2:]:
                    out.append(val[2:])
            else:
                for elem in val:
                    if isinstance(elem, list):
                        if len(elem) == 1:
                            out.append(elem[0])
                        else:
                            out.append(self.clean(elem))
                    else:
                        out.append(elem)
            val = out
        return out[0]

    def title(self, value):
        if self.is_title:
            out = []
            value = value.split()
            for word in value:
                if word in self.exclude_title:
                    out.append(word)
                else:
                    out.append(word[0].upper() + word[1:])
            value = " ".join(out)
        return value

    def verify_ordinal(self, value):
        if not value == int(value):
            raise TypeError(self.errmsg_floatord % value)
        if not abs(value) == value:
            raise TypeError(self.errmsg_negord % value)

    def to_ordinal(self, value):
        return self.to_cardinal(value)

    def to_ordinal_num(self, value):
        return value

    # Trivial version
    def inflect(self, value, text):
        text = text.split("/")
        if value == 1:
            return text[0]
        return "".join(text)

    # //CHECK: generalise? Any others like pounds/shillings/pence?
    def to_splitnum(self, val, hightxt="", lowtxt="", jointxt="",
                    divisor=100, longval=True, cents=True):
        out = []

        if isinstance(val, float):
            high, low = self.float2tuple(val)
        else:
            try:
                high, low = val
            except TypeError:
                high, low = divmod(val, divisor)

        if high:
            hightxt = self.title(self.inflect(high, hightxt))
            out.append(self.to_cardinal(high))
            if low:
                if longval:
                    if hightxt:
                        out.append(hightxt)
                    if jointxt:
                        out.append(self.title(jointxt))
            elif hightxt:
                out.append(hightxt)

        if low:
            if cents:
                out.append(self.to_cardinal(low))
            else:
                out.append("%02d" % low)
            if lowtxt and longval:
                out.append(self.title(self.inflect(low, lowtxt)))

        return " ".join(out)

    def to_year(self, value, **kwargs):
        return self.to_cardinal(value)

    def pluralize(self, n, forms):
        """
        Should resolve gettext form:
        http://docs.translatehouse.org/projects/localization-guide/en/latest/l10n/pluralforms.html
        """
        raise NotImplementedError

    def _money_verbose(self, number, currency):
        return self.to_cardinal(number)

    def _cents_verbose(self, number, currency):
        return self.to_cardinal(number)

    def _cents_terse(self, number, currency):
        return "%02d" % number

    def to_currency(self, val, currency='EUR', cents=True, separator=',',
                    adjective=False):
        """
        Args:
            val: Numeric value
            currency (str): Currency code
            cents (bool): Verbose cents
            separator (str): Cent separator
            adjective (bool): Prefix currency name with adjective
        Returns:
            str: Formatted string

        Handles whole numbers and decimal numbers differently
        """
        left, right, is_negative = parse_currency_parts(val)

        try:
            cr1, cr2 = self.CURRENCY_FORMS[currency]

        except KeyError:
            raise NotImplementedError(
                'Currency code "%s" not implemented for "%s"' %
                (currency, self.__class__.__name__))

        if adjective and currency in self.CURRENCY_ADJECTIVES:
            cr1 = prefix_currency(self.CURRENCY_ADJECTIVES[currency], cr1)

        minus_str = "%s " % self.negword.strip() if is_negative else ""
        money_str = self._money_verbose(left, currency)

        # Explicitly check if input has decimal point or non-zero cents
        has_decimal = isinstance(val, float) or str(val).find('.') != -1

        # Only include cents if:
        # 1. Input has decimal point OR
        # 2. Cents are non-zero
        if has_decimal or right > 0:
            cents_str = self._cents_verbose(right, currency) \
                if cents else self._cents_terse(right, currency)

            return u'%s%s %s%s %s %s' % (
                minus_str,
                money_str,
                self.pluralize(left, cr1),
                separator,
                cents_str,
                self.pluralize(right, cr2)
            )
        else:
            return u'%s%s %s' % (
                minus_str,
                money_str,
                self.pluralize(left, cr1)
            )

    def setup(self):
        pass

GENERIC_DOLLARS = ('dollar', 'dollars')
GENERIC_CENTS = ('cent', 'cents')


class Num2Word_EU(Num2Word_Base):
    CURRENCY_FORMS = {
        'AUD': (GENERIC_DOLLARS, GENERIC_CENTS),
        'BYN': (('rouble', 'roubles'), ('kopek', 'kopeks')),
        'CAD': (GENERIC_DOLLARS, GENERIC_CENTS),
        # repalced by EUR
        'EEK': (('kroon', 'kroons'), ('sent', 'senti')),
        'EUR': (('euro', 'euro'), GENERIC_CENTS),
        'GBP': (('pound sterling', 'pounds sterling'), ('penny', 'pence')),
        # replaced by EUR
        'LTL': (('litas', 'litas'), GENERIC_CENTS),
        # replaced by EUR
        'LVL': (('lat', 'lats'), ('santim', 'santims')),
        'USD': (GENERIC_DOLLARS, GENERIC_CENTS),
        'RUB': (('rouble', 'roubles'), ('kopek', 'kopeks')),
        'SEK': (('krona', 'kronor'), ('öre', 'öre')),
        'NOK': (('krone', 'kroner'), ('øre', 'øre')),
        'PLN': (('zloty', 'zlotys', 'zlotu'), ('grosz', 'groszy')),
        'MXN': (('peso', 'pesos'), GENERIC_CENTS),
        'RON': (('leu', 'lei', 'de lei'), ('ban', 'bani', 'de bani')),
        'INR': (('rupee', 'rupees'), ('paisa', 'paise')),
        'HUF': (('forint', 'forint'), ('fillér', 'fillér')),
        'ISK': (('króna', 'krónur'), ('aur', 'aurar')),
        'UZS': (('sum', 'sums'), ('tiyin', 'tiyins')),
        'SAR': (('saudi riyal', 'saudi riyals'), ('halalah', 'halalas')),
        'JPY': (('yen', 'yen'), ('sen', 'sen')),
        'KRW': (('won', 'won'), ('jeon', 'jeon')),

    }

    CURRENCY_ADJECTIVES = {
        'AUD': 'Australian',
        'BYN': 'Belarusian',
        'CAD': 'Canadian',
        'EEK': 'Estonian',
        'USD': 'US',
        'RUB': 'Russian',
        'NOK': 'Norwegian',
        'MXN': 'Mexican',
        'RON': 'Romanian',
        'INR': 'Indian',
        'HUF': 'Hungarian',
        'ISK': 'íslenskar',
        'UZS': 'Uzbekistan',
        'SAR': 'Saudi',
        'JPY': 'Japanese',
        'KRW': 'Korean',
    }

    GIGA_SUFFIX = "illiard"
    MEGA_SUFFIX = "illion"

    def set_high_numwords(self, high):
        cap = 3 + 6 * len(high)

        for word, n in zip(high, range(cap, 3, -6)):
            if self.GIGA_SUFFIX:
                self.cards[10 ** n] = word + self.GIGA_SUFFIX

            if self.MEGA_SUFFIX:
                self.cards[10 ** (n - 3)] = word + self.MEGA_SUFFIX

    def gen_high_numwords(self, units, tens, lows):
        out = [u + t for t in tens for u in units]
        out.reverse()
        return out + lows

    def pluralize(self, n, forms):
        form = 0 if n == 1 else 1
        return forms[form]

    def setup(self):
        lows = ["non", "oct", "sept", "sext", "quint", "quadr", "tr", "b", "m"]
        units = ["", "un", "duo", "tre", "quattuor", "quin", "sex", "sept",
                 "octo", "novem"]
        tens = ["dec", "vigint", "trigint", "quadragint", "quinquagint",
                "sexagint", "septuagint", "octogint", "nonagint"]
        self.high_numwords = ["cent"] + self.gen_high_numwords(units, tens,
                                                               lows)

ZERO = 'nulla'


class Num2Word_HU(Num2Word_EU):
    GIGA_SUFFIX = "illiárd"
    MEGA_SUFFIX = "illió"

    def setup(self):
        super(Num2Word_HU, self).setup()

        self.negword = "mínusz "
        self.pointword = "egész"

        self.mid_numwords = [(1000, "ezer"), (100, "száz"), (90, "kilencven"),
                             (80, "nyolcvan"), (70, "hetven"), (60, "hatvan"),
                             (50, "ötven"), (40, "negyven"), (30, "harminc")]

        low_numwords = ["kilenc", "nyolc", "hét", "hat", "öt", "négy", "három",
                        "kettő", "egy"]
        self.low_numwords = (['tizen' + w for w in low_numwords]
                             + ['tíz']
                             + low_numwords)
        self.low_numwords = (['huszon' + w for w in low_numwords]
                             + ['húsz']
                             + self.low_numwords
                             + [ZERO])

        self.partial_ords = {
            'nulla': 'nullad',
            'egy': 'egyed',
            'kettő': 'ketted',
            'három': 'harmad',
            'négy': 'negyed',
            'öt': 'ötöd',
            'hat': 'hatod',
            'hét': 'heted',
            'nyolc': 'nyolcad',
            'kilenc': 'kilenced',
            'tíz': 'tized',
            'húsz': 'huszad',
            'harminc': 'harmincad',
            'negyven': 'negyvened',
            'ötven': 'ötvened',
            'hatvan': 'hatvanad',
            'hetven': 'hetvened',
            'nyolcvan': 'nyolcvanad',
            'kilencven': 'kilencvened',
            'száz': 'század',
            'ezer': 'ezred',
            'illió': 'milliomod',
            'illiárd': 'milliárdod'
        }

    def to_cardinal(self, value, zero=ZERO):
        if int(value) != value:
            return self.to_cardinal_float(value)
        elif value < 0:
            out = self.negword + self.to_cardinal(-value)
        elif value == 0:
            out = zero
        elif zero == '' and value == 2:
            out = 'két'
        elif value < 30:
            out = self.cards[value]
        elif value < 100:
            out = self.tens_to_cardinal(value)
        elif value < 1000:
            out = self.hundreds_to_cardinal(value)
        elif value < 10**6:
            out = self.thousands_to_cardinal(value)
        else:
            out = self.big_number_to_cardinal(value)
        return out

    def tens_to_cardinal(self, value):
        try:
            return self.cards[value]
        except KeyError:
            return self.cards[value // 10 * 10] + self.to_cardinal(value % 10)

    def hundreds_to_cardinal(self, value):
        hundreds = value // 100
        prefix = "száz"
        if hundreds != 1:
            prefix = self.to_cardinal(hundreds, zero="") + prefix
        postfix = self.to_cardinal(value % 100, zero="")
        return prefix + postfix

    def thousands_to_cardinal(self, value):
        thousands = value // 1000
        prefix = "ezer"
        if thousands != 1:
            prefix = self.to_cardinal(thousands, zero="") + prefix
        postfix = self.to_cardinal(value % 1000, zero="")
        return prefix + ('' if value <= 2000 or not postfix else '-') + postfix

    def big_number_to_cardinal(self, value):
        digits = len(str(value))
        digits = digits if digits % 3 != 0 else digits - 2
        exp = 10 ** (digits // 3 * 3)
        rest = self.to_cardinal(value % exp, '')
        return (self.to_cardinal(value // exp, '') + self.cards[exp]
                + ('-' + rest if rest else ''))

    def to_ordinal(self, value):
        if value < 0:
            return self.negword + self.to_ordinal(-value)
        if value == 1:
            return 'első'
        elif value == 2:
            return 'második'
        else:
            out = self.to_cardinal(value)
            for card_word, ord_word in self.partial_ords.items():
                if out[-len(card_word):] == card_word:
                    out = out[:-len(card_word)] + ord_word
                    break
        return out + 'ik'

    def to_ordinal_num(self, value):
        self.verify_ordinal(value)
        return str(value) + '.'

    def to_year(self, val, suffix=None, longval=True):
        # suffix is prefix here
        prefix = ''
        if val < 0 or suffix is not None:
            val = abs(val)
            prefix = (suffix + ' ' if suffix is not None else 'i. e. ')
        return prefix + self.to_cardinal(val)

    def to_currency(self, val, currency='HUF', cents=True, separator=',',
                    adjective=False):
        return super(Num2Word_HU, self).to_currency(
            val, currency, cents, separator, adjective)

    def to_cardinal_float(self, value):
        if abs(value) != value:
            return self.negword + self.to_cardinal_float(-value)
        left, right = str(value).split('.')
        return (self.to_cardinal(int(left))
                + ' egész '
                + self.to_cardinal(int(right))
                + ' ' + self.partial_ords[self.cards[10 ** len(right)]])

# A num2words függvény, amit a normaliser.py használni fog
def num2words(number, to='cardinal', lang='hu', **kwargs):
    converter = Num2Word_HU()
    if to == 'cardinal':
        return converter.to_cardinal(number, **kwargs)
    elif to == 'ordinal':
        return converter.to_ordinal(number, **kwargs)
    elif to == 'ordinal_num':
        return converter.to_ordinal_num(number, **kwargs)
    elif to == 'year':
        return converter.to_year(number, **kwargs)
    elif to == 'currency':
        return converter.to_currency(number, **kwargs)
    else:
        raise NotImplementedError("A 'to' paraméter csak 'cardinal', 'ordinal', 'ordinal_num', 'year' vagy 'currency' lehet.")
