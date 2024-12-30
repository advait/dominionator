use std::fmt::Display;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Card {
    pub name: &'static str,
    pub short_name: &'static str,
    pub cost: u8,
    pub treasure: u8,
    pub victory: u8,
}

impl Display for Card {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.short_name)
    }
}

pub static COPPER: Card = Card {
    name: "Copper",
    short_name: "C",
    cost: 0,
    treasure: 1,
    victory: 0,
};

pub static SILVER: Card = Card {
    name: "Silver",
    short_name: "S",
    cost: 3,
    treasure: 2,
    victory: 0,
};

pub static GOLD: Card = Card {
    name: "Gold",
    short_name: "G",
    cost: 6,
    treasure: 3,
    victory: 0,
};

pub static ESTATE: Card = Card {
    name: "Estate",
    short_name: "E",
    cost: 2,
    treasure: 0,
    victory: 1,
};

pub static DUCHY: Card = Card {
    name: "Duchy",
    short_name: "D",
    cost: 5,
    treasure: 0,
    victory: 3,
};

pub static PROVINCE: Card = Card {
    name: "Province",
    short_name: "P",
    cost: 8,
    treasure: 0,
    victory: 6,
};

pub const N_CARDS: usize = 6;
pub static CARD_SET: [&Card; N_CARDS] = [&COPPER, &SILVER, &GOLD, &ESTATE, &DUCHY, &PROVINCE];

pub static DEFAULT_DECK: [&Card; 10] = [
    &COPPER, &COPPER, &COPPER, &COPPER, &COPPER, &COPPER, &COPPER, &ESTATE, &ESTATE, &ESTATE,
];

pub static DEFAULT_KINGDOM: [(&Card, u8); 6] = [
    (&COPPER, 60),
    (&SILVER, 40),
    (&GOLD, 30),
    (&ESTATE, 8),
    (&DUCHY, 10),
    (&PROVINCE, 12),
];
