#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Card {
    pub name: &'static str,
    pub short_name: &'static str,
    pub cost: u8,
    pub treasure: u8,
    pub victory: u8,
}

pub static COPPER: Card = Card {
    name: "Copper",
    short_name: "C",
    cost: 0,
    treasure: 1,
    victory: 0,
};

pub static ESTATE: Card = Card {
    name: "Estate",
    short_name: "E",
    cost: 2,
    treasure: 0,
    victory: 1,
};

pub const N_CARDS: usize = 2;
pub static CARD_SET: [&Card; N_CARDS] = [&COPPER, &ESTATE];

pub static DEFAULT_DECK: [&Card; 10] = [
    &COPPER, &COPPER, &COPPER, &COPPER, &COPPER, &COPPER, &COPPER, &ESTATE, &ESTATE, &ESTATE,
];

pub static DEFAULT_KINGDOM: [(&Card, u8); 2] = [(&COPPER, 60), (&ESTATE, 12)];
