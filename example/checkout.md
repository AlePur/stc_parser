# CheckoutService

_Auto-generated from `checkout.stc` by the STC compiler._

```mermaid
stateDiagram-v2
    Reserved : steps(ReserveStock)
    Processing : steps(ChargeCard[locked])
    Completed : steps(Finish)

    [*] --> Created
    Created --> Reserved : success
    Reserved --> Processing : success
    Reserved --> Cancelled : fail
    Processing --> Completed : success
    Processing --> PaymentFailed : fail
    Completed --> [*]
    Cancelled --> [*]
    PaymentFailed --> [*]
```
